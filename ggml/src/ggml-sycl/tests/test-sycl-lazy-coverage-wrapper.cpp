//
// The wrapper a planner caller holds around its planning (th32 §3.2, §3.3, §11 T3): lazy_coverage_scope runs its
// entry FULL reap before it takes the provisional lease, and its closing FULL reap after it drops that lease, so a
// planner never pins a pool another reaper is about to free and the scope never leaves a parked pool behind.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The scope is constructed directly, with no dry run and no stage, since neither adopts it before the IO
// engine does. The pool registry lives in the backend's objects, so the binary links the private fixtures and is
// registered gpu-serial and serial like the other lazy-table tests; it initialises no device.
//
// Only a take-time snapshot can score the entry reap: the final totals read the same whether the entry reap ran before
// the take, after it, or was replaced by a later reap. So the fixture parks a node first, holds that node's own reap in
// the hook before its list swap, and only then constructs the scope: the entry reap is the one that frees the node, and
// the take that follows it must find the reaps run one past the held reap's and no node parked.
//
// Every wait is bounded, so an implementation whose reap never reaches the hook fails here instead of hanging.

#include "../lazy-io-pool.hpp"
#include "host-ledger-test-harness.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace ggml_sycl;
using namespace host_ledger_test_harness;

namespace {

constexpr std::chrono::seconds k_hook_bound{ 10 };

// The hold hook's rendezvous: the held reap reports itself and blocks until the fixture releases it.
struct reap_hold {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
};

// Lets a held reap go, and every later call of the hook return at once.
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

lazy_registry_state registry() {
    lazy_registry_state state;
    if (!ggml_sycl_lazy_registry_for_testing(&state)) {
        std::printf("FAIL: the registry accessor refused a valid pointer\n");
    }
    return state;
}

// The wrapper arm. A helper guard's pool is parked by the guard's own last drop, and that drop's FULL reap is held in
// the hook after its count lands and before it swaps the list out, so the node is still parked. The fixture reads c0,
// the FULL reaps run, then constructs the scope with take_if(true): the scope's entry reap is the one that frees the
// node, so the take after it snapshots c0 + 1 reaps run and 0 parked nodes. Destroying the scope drops its lease, which
// parks the scope's pool, and its closing reap frees it: no node parked and c0 + 2 reaps run.
// Engagement: the guard holds a lease on a live pool, its drop parks that pool, and the hook reports the held reap
// within the bound with one node still parked.
// Mutants: the entry reap dropped (the take snapshots c0 reaps and the parked node), or run after the take (the take
// snapshots the parked node); the closing reap dropped (the scope's pool survives it parked); the scope taking no lease
// for a lazy candidate (take_if false, no pin).
arm_result arm_entry_and_closing_reaps() {
    arm_result r;
    reap_hold  hold;

    std::unique_ptr<lazy_test_coverage_pin> guard(new lazy_test_coverage_pin(ggml_sycl_test_make_coverage_pin(0)));
    const lazy_registry_state               before = registry();
    expect(r, guard->pin() != nullptr, "the helper guard took no lease: the registry built no pool to pin");
    expect(r, before.live_pools == 1 && before.live_leases == 1 && before.parked_nodes == 0,
           "with the guard alone the registry holds " + num(before.live_pools) + " pools, " + num(before.live_leases) +
               " leases and " + num(before.parked_nodes) + " parked nodes, want 1, 1 and 0");

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
        fail(r, "the guard's drop never reached the hold hook within " + num(k_hook_bound.count()) +
                    " s: its last drop parked no pool, or the FULL reap after it never ran");
    } else {
        expect(r, held.parked_nodes == 1 && held.live_pools == 0,
               "while the guard's reap is held the registry has " + num(held.parked_nodes) + " parked nodes and " +
                   num(held.live_pools) + " live pools, want the guard's pool parked and none live");
    }

    {
        lazy_coverage_scope       scope;
        const bool                took  = scope.take_if(true);
        const lazy_registry_state taken = registry();
        expect(r, took && scope.pin() != nullptr, "take_if(true) took no lease for a lazy candidate");
        expect(r, taken.reaps_run_at_last_take == c0 + 1,
               "the take snapshot " + num(taken.reaps_run_at_last_take) + " FULL reaps run, want " + num(c0 + 1) +
                   ": the scope's entry reap did not run before its take");
        expect(r, taken.parked_at_last_take == 0,
               "the take found " + num(taken.parked_at_last_take) +
                   " parked nodes: the entry reap did not free the parked pool before the take pinned a new one");

        release(hold);
        dropper.join();
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
    }

    const lazy_registry_state after = registry();
    expect(r, after.parked_nodes == 0,
           "the scope left " + num(after.parked_nodes) + " parked nodes: its closing reap did not free its own pool");
    expect(r, after.full_reaps_run == c0 + 2,
           "after the scope " + num(after.full_reaps_run) + " FULL reaps have run, want " + num(c0 + 2) +
               " (the held reap, the entry reap and the closing reap)");
    expect(r, after.live_pools == 0 && after.live_leases == 0,
           "after the scope the registry holds " + num(after.live_pools) + " pools and " + num(after.live_leases) +
               " leases, want none");
    return r;
}

// take_if(false), an inventory with no lazy candidate, takes nothing, and the scope still runs both reaps.
// Engagement: the scope is constructed and destroyed with the hook disarmed.
// Mutants: take_if taking a lease regardless of the candidate (a pin and a live pool); either reap gated on the take
// (fewer than two reaps run).
arm_result arm_no_candidate_takes_nothing() {
    arm_result                r;
    const lazy_registry_state start = registry();
    {
        lazy_coverage_scope scope;
        expect(r, !scope.take_if(false), "take_if(false) reported a lease for an inventory with no lazy candidate");
        expect(r, scope.pin() == nullptr, "the scope hands the planner a pin with no lazy candidate");
        expect(r, registry().live_pools == 0, "take_if(false) built a pool");
    }
    const lazy_registry_state end = registry();
    expect(r, end.full_reaps_run == start.full_reaps_run + 2,
           "a scope with no candidate ran " + num(end.full_reaps_run - start.full_reaps_run) +
               " FULL reaps, want 2: the entry and the closing reap do not depend on the take");
    return r;
}

const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

}  // namespace

int main() {
    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "the entry reap runs before the take and the closing reap after the drop", arm_entry_and_closing_reaps    },
        { "a scope with no lazy candidate takes nothing and still reaps twice",      arm_no_candidate_takes_nothing },
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
    std::printf("lazy coverage wrapper arms: %d passed, %d failed, %d void\n", passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
