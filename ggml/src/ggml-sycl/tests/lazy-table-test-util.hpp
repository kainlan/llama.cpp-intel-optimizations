//
// What the lazy-table tests share (th32 §11 T3): the constants their thresholds derive from, the pool registry's
// reading, the hold hook's rendezvous, the coverage wrapper's two arms, the --devices option and the arm loop. The pool
// host test links lazy-io-pool.cpp alone, so nothing here reaches a device, the unified cache or a SYCL header. Each
// test includes it once.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "../lazy-io-pool.hpp"
#include "host-ledger-test-harness.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lazy_table_test {

using host_ledger_test_harness::arm_result;
using host_ledger_test_harness::expect;
using host_ledger_test_harness::fail;
using host_ledger_test_harness::num;
using host_ledger_test_harness::outcome;

constexpr uint64_t MiB = 1024ull * 1024ull;
constexpr uint64_t GiB = 1024ull * MiB;

// P3's margin over expected_host: 24 GiB left after load, plus P3's 5 GiB runtime allowance (§4.3 step 4).
constexpr uint64_t k_p3_margin = 29 * GiB;

// How long a fixture waits for a held reap to reach the hold hook, so a reap that never reaches it fails the arm
// instead of hanging it.
constexpr std::chrono::seconds k_hook_bound{ 10 };

// --- the pool registry ---

inline ggml_sycl::lazy_registry_state registry() {
    ggml_sycl::lazy_registry_state state;
    if (!ggml_sycl::ggml_sycl_lazy_registry_for_testing(&state)) {
        std::printf("FAIL: the registry accessor refused a valid pointer\n");
    }
    return state;
}

inline std::string describe(const ggml_sycl::lazy_registry_state & s) {
    return num(s.live_pools) + " live pools, " + num(s.live_leases) + " leases, " + num(s.parked_nodes) + " parked";
}

inline bool registry_is(const ggml_sycl::lazy_registry_state & s, uint64_t pools, uint64_t leases, uint64_t parked) {
    return s.live_pools == pools && s.live_leases == leases && s.parked_nodes == parked;
}

// --- the hold hook's rendezvous ---

// The held reap reports itself and blocks until the fixture releases it.
struct reap_hold {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    entered  = false;
    bool                    released = false;
};

// Lets a held reap go, and every later call of the hook return at once.
inline void release(reap_hold & hold) {
    {
        std::lock_guard<std::mutex> lock(hold.mutex);
        hold.released = true;
    }
    hold.cv.notify_all();
}

inline void hold_reap(void * ctx) {
    reap_hold *                  hold = static_cast<reap_hold *>(ctx);
    std::unique_lock<std::mutex> lock(hold->mutex);
    hold->entered = true;
    hold->cv.notify_all();
    hold->cv.wait(lock, [hold] { return hold->released; });
}

// --- the coverage wrapper's arms (§3.3, §11 T3's wrapper arm) ---

// A helper guard's pool is parked by the guard's own last drop, and that drop's FULL reap is held in the hook after its
// count lands and before it swaps the list out, so the node is still parked. The fixture reads c0, the FULL reaps run,
// then constructs the scope with take_if(true): the scope's entry reap is the one that frees the node, so the take
// after it snapshots c0 + 1 reaps run and no parked node. Destroying the scope drops its lease, which parks the scope's
// pool, and its closing reap frees it: no node parked and c0 + 2 reaps run.
// Engagement: the guard holds a lease on a live pool, its drop parks that pool, and the hook reports the held reap
// within the bound with one node still parked.
// Mutants: the entry reap dropped (the take snapshots c0 reaps and the parked node), or run after the take (the take
// snapshots the parked node); the closing reap dropped (the scope's pool survives it parked); the scope taking no lease
// for a lazy candidate. The two dropped reaps are the pool host test's registered controls,
// LAZY_IO_POOL_MUTANT_NO_ENTRY_REAP and LAZY_IO_POOL_MUTANT_NO_CLOSING_REAP, whose PASS_REGULAR_EXPRESSIONs match this
// arm's name and its two FAIL texts.
inline arm_result arm_wrapper_reaps() {
    arm_result r;
    reap_hold  hold;

    std::unique_ptr<ggml_sycl::lazy_test_coverage_pin> guard(
        new ggml_sycl::lazy_test_coverage_pin(ggml_sycl::ggml_sycl_test_make_coverage_pin(0)));
    const ggml_sycl::lazy_registry_state before = registry();
    expect(r, guard->pin() != nullptr, "the helper guard took no lease: the registry built no pool to pin");
    expect(r, registry_is(before, 1, 1, 0), "with the guard alone: " + describe(before) + ", want 1, 1, 0");

    ggml_sycl::ggml_sycl_test_lazy_reap_hold(hold_reap, &hold);
    std::thread dropper([&guard] { guard.reset(); });
    bool        entered = false;
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        entered = hold.cv.wait_for(lock, k_hook_bound, [&hold] { return hold.entered; });
    }
    const ggml_sycl::lazy_registry_state held = registry();
    const uint64_t                       c0   = held.full_reaps_run;
    if (!entered) {
        // Released and disarmed before the scope exists, so a late reap cannot hold the scope's own entry reap.
        release(hold);
        ggml_sycl::ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
        fail(r, "the guard's drop never reached the hold hook within " + num(k_hook_bound.count()) +
                    " s: its last drop parked no pool, or the FULL reap after it never ran");
    } else {
        expect(r, registry_is(held, 0, 0, 1), "while the guard's reap is held: " + describe(held) + ", want 0, 0, 1");
    }

    {
        ggml_sycl::lazy_coverage_scope       scope;
        const bool                           took  = scope.take_if(true);
        const ggml_sycl::lazy_registry_state taken = registry();
        expect(r, took && scope.pin() != nullptr, "take_if(true) took no lease for a lazy candidate");
        expect(r, taken.reaps_run_at_last_take == c0 + 1,
               "the take snapshot " + num(taken.reaps_run_at_last_take) + " reaps run, want " + num(c0 + 1) +
                   ": the entry reap did not run before the take");
        expect(r, taken.parked_at_last_take == 0,
               "the take found " + num(taken.parked_at_last_take) +
                   " parked nodes: the entry reap did not free the parked pool before the take pinned a new one");

        release(hold);
        dropper.join();
        ggml_sycl::ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
    }

    const ggml_sycl::lazy_registry_state after = registry();
    expect(r, registry_is(after, 0, 0, 0),
           "after the scope: " + describe(after) + ", want its pool reaped by its closing reap");
    expect(r, after.full_reaps_run == c0 + 2,
           "after the scope " + num(after.full_reaps_run) + " FULL reaps have run, want " + num(c0 + 2) +
               " (the held reap, the entry reap and the closing reap)");
    return r;
}

// take_if(false), an inventory with no lazy candidate, takes nothing, and the scope still runs both reaps.
// Engagement: the scope is constructed and destroyed with the hook disarmed.
// Mutants: take_if taking a lease regardless of the candidate (a pin and a live pool); either reap gated on the take
// (fewer than two reaps run).
inline arm_result arm_wrapper_no_candidate() {
    arm_result                           r;
    const ggml_sycl::lazy_registry_state start = registry();
    {
        ggml_sycl::lazy_coverage_scope scope;
        expect(r, !scope.take_if(false), "take_if(false) reported a lease for an inventory with no lazy candidate");
        expect(r, scope.pin() == nullptr, "the scope hands the planner a pin with no lazy candidate");
        expect(r, registry().live_pools == 0, "take_if(false) built a pool");
    }
    const ggml_sycl::lazy_registry_state end = registry();
    expect(r, end.full_reaps_run == start.full_reaps_run + 2,
           "a scope with no candidate ran " + num(end.full_reaps_run - start.full_reaps_run) +
               " FULL reaps, want 2: the entry and the closing reap do not depend on the take");
    return r;
}

// --- the arm loop ---

struct named_arm {
    const char * name;
    arm_result (*run)();
};

// The wrapper arms under the names both tests print; the pool host test's registered controls match the first name.
inline std::vector<named_arm> wrapper_arms() {
    return {
        { "the wrapper reaps before its take and after its drop",               arm_wrapper_reaps        },
        { "a scope with no lazy candidate takes nothing and still reaps twice", arm_wrapper_no_candidate },
    };
}

inline const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

// Runs every arm, so one failure does not hide the rest, printing each verdict, then `label` with the counts. Returns
// the binary's status: 0 only when every arm passed, since a VOID arm fails the binary like a FAIL.
inline int run_arms(const std::vector<named_arm> & arms, const std::string & label) {
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
    std::printf("%s: %d passed, %d failed, %d void\n", label.c_str(), passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}

// --- the --devices option ---

// Reads --devices N into `devices`, 1 when the option is absent. False, after a FAIL line, unless it is 1 or 2.
inline bool parse_devices(int argc, char ** argv, int & devices) {
    devices = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            devices = std::atoi(argv[++i]);
        }
    }
    if (devices != 1 && devices != 2) {
        std::printf("FAIL: --devices must be 1 or 2, got %d\n", devices);
        return false;
    }
    return true;
}

}  // namespace lazy_table_test
