//
// The IO pool's reap points (th32 §3.3, §11 T5), one fixture per point class, and the race between a lease's last drop
// and a take: the entry reap of a planner caller, the reap after a failed load's end, the teardown reap at a model
// swap and the teardown reap whose first step throws, the shutdown drain with a read in flight, with the last table and
// read freed and with a published record never erased, and the two orders of a read's last drop against a stage's
// take. The reader-thread point is the engine host test's worker-drop arm.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. The stage initialises the SYCL backends, constructs their caches and plans through them, so the binary is
// registered gpu-serial, serial, with the selector pinned to both discrete cards (level_zero:0,1). It runs iff the
// planner caller takes its multi-device path, ggml_backend_sycl_moe_multi_gpu_requested(), and exits 77 otherwise. No
// libllama: the fixture builds each inventory and drives the load transaction itself, and the engine's reads come from
// the PRIVATE_TESTING read source. The teardown throw, the three shutdown fixtures and their mutants run in children
// of this binary, started before this process initialises SYCL, since a shutdown ends the backend for the process and
// a throw that escapes a noexcept reap ends it.
//
// --mutant <name> selects one runtime mutant of the PRIVATE_TESTING selector (§11) and runs only the arm that kills it;
// each --mutant registration passes only when that arm prints its FAIL line. An arm whose engagement check fails is
// VOID, which fails the binary like a FAIL. Every in-process arm ends with a FULL reap that must leave no pool.

#include "../lazy-io-pool.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-plan-shaped-inventory.hpp"
#include "lazy-ring-test-fixture.hpp"
#include "sycl-test-skip.hpp"

#include <signal.h>
#include <sys/wait.h>

#include <atomic>
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

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using namespace lazy_table_test;
using namespace lazy_ring_fixture;
using host_plan_shaped_inventory::loader_inventory;

namespace {

// §11 T5's bound for a wait on an observed event, and the bound for a child that initialises the backends and loads.
constexpr std::chrono::seconds      k_bound{ 5 };
constexpr std::chrono::milliseconds k_child_bound{ 180000 };

const char * k_self = "test-sycl-lazy-reap-points";

// The arms' names, which the mutant registrations' PASS_REGULAR_EXPRESSIONs match.
const char * k_arm_entry = "a stage's entry reap frees a ring another thread parked before the stage takes its lease";
const char * k_arm_failed_load =
    "a load that ends failed after planning STORAGE gives its ring back before its end returns";
const char * k_arm_swap  = "a model swap gives A's ring back at A's teardown with no witnessed lock held";
const char * k_arm_throw = "a teardown reap whose first step throws logs it and leaves the unload finalized once";
const char * k_arm_shutdown_read = "shutdown with a read in flight returns after the read's drop and leaves no pool";
const char * k_arm_shutdown_freed =
    "with the last table and read freed the ring's bytes return and shutdown leaves no pool";
const char * k_arm_leaked = "shutdown drains a never-erased record's lease with one leaked-lease line";
const char * k_arm_drop_first =
    "a read's last drop between the stage's entry reap and its take parks, and the take builds anew";
const char * k_arm_take_first = "a read's drop held while the stage takes leaves the pool live and its ring unchanged";

// The texts the children print and the arms score.
const char * k_reap_failed  = "[HOST-PLAN-BUG] ring reap failed: the test seam's throw at the reap's first step";
const char * k_lease_locked = "[HOST-PLAN-BUG] ring lease dropped with";
const char * k_leaked       = "[HOST-PLAN-BUG] leaked ring lease: model=";

// The mutant a child plants, passed on its command line; null in a plain run.
const char * g_child_mutant = nullptr;

// --- small readings ---

std::string status_text(int status) {
    if (WIFSIGNALED(status)) {
        return "signal " + std::to_string(WTERMSIG(status));
    }
    return "exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

// Polls `done` until it holds or 5 s pass: an observation, never a sleep that stands in for an event.
template <typename F> bool wait_bounded(F done) {
    const auto deadline = std::chrono::steady_clock::now() + k_bound;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// A gate a hook's holder blocks on until the arm releases it; the arm waits at most 5 s for the hold.
struct bounded_gate {
    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  holders  = 0;
    bool                    released = false;

    void hold() {
        std::unique_lock<std::mutex> lock(mutex);
        ++holders;
        cv.notify_all();
        cv.wait(lock, [this] { return released; });
    }

    bool wait_holders(size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, k_bound, [this, n] { return holders >= n; });
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

// The first captured log line that contains `needle`, or "(none)".
std::string first_log_line_with(const char * needle) {
    const std::vector<std::string> lines = log_lines_containing(needle);
    return lines.empty() ? std::string("(none)") : chomp(lines.front());
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

// The live ring's bytes through the published accessor, or 0 when it reads none.
uint64_t live_ring() {
    uint64_t v = 0;
    return lazy_block_ring_bytes(&v) ? v : 0;
}

// A pool whose only lease is a stand-in for an in-flight read, held by this thread: the coverage pin ensures the ring
// on device 0, the read joins it, and the pin's drop leaves the read's lease alone.
ring_lease read_lease_on_live_ring(arm_result & r) {
    pin_ptr    pin  = make_pin(k_need_8k);
    ring_lease read = lazy_io_pool_join();
    pin.reset();
    const lazy_registry_state s = registry();
    if (!read.held() || !registry_is(s, 1, 1, 0) || s.ring_bytes != k_need_8k) {
        void_arm(r, "the stand-in read holds no lease on a live ring of " + num(k_need_8k) + " B: " + describe(s));
    }
    return read;
}

// --- the entry reap ---

// What the stage-take hook read at model B's first take, after its entry reap: the parked nodes.
struct take_view {
    bool     entered = false;
    uint64_t parked  = 0;
};

void read_take_view(void * ctx) {
    take_view * v = static_cast<take_view *>(ctx);
    v->entered    = true;
    v->parked     = registry().parked_nodes;
}

// The entry reap (§3.3 point (i)): a ring parked by a test thread, never a pool worker, whose own FULL reap the hook
// holds after its count lands, is freed by the next stage's entry reap before that stage's provisional take. So model
// B's first take runs with no parked node, B plans ring= the mapped need on both passes, and B's ensure allocates
// exactly the need: the ledger shows one ring.
// The parked count is read at the first take itself: the early pass's closing reap frees the parked ring whether or not
// the entry reap ran, so the late take, the plans and the ledger after B's ensure read the same either way.
// Engagement, VOID counting as failed: the dropper's reap is held within 5 s with the node parked, and B's stage
// reaches its take.
// Mutant: no-entry-reap, under which B's first take runs with A's ring still parked.
arm_result arm_entry() {
    arm_result     r;
    lazy_inventory inv_b("blk.1.entry_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    ring_lease     read  = read_lease_on_live_ring(r);
    if (r.result != outcome::PASS) {
        read.reset();
        return settle(r, role0);
    }
    reap_hold hold;
    ggml_sycl_test_lazy_reap_hold(hold_reap, &hold);
    std::thread dropper([&read] {
        read.reset();  // the last drop: parks the pool with its ring
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    });
    bool        entered = false;
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        entered = hold.cv.wait_for(lock, k_bound, [&hold] { return hold.entered; });
    }
    const lazy_registry_state parked = registry();
    if (!entered || parked.parked_nodes != 1) {
        release(hold);
        dropper.join();
        ggml_sycl_test_lazy_reap_hold(nullptr, nullptr);
        void_arm(r, std::string("the dropper's reap ") + (entered ? "was held" : "was not held within 5 s") + " with " +
                        describe(parked) + ", want 1 parked");
        return settle(r, role0);
    }
    take_view view;
    ggml_sycl_test_lazy_stage_take_hold(read_take_view, &view);
    model_load b = load_model(r, inv_b, "model B");
    ggml_sycl_test_lazy_stage_take_hold(nullptr, nullptr);
    release(hold);
    dropper.join();
    if (!b.ok) {
        return settle(r, role0);
    }
    if (!view.entered) {
        void_arm(r, "model B's stage did not reach its take after the entry reap");
    }
    expect(r, view.parked == 0,
           "B's first take ran with " + num(view.parked) +
               " parked node(s) after its entry reap, want 0: the entry reap left the ring parked");
    const lazy_ring_ensure_status st = ensure_for(b);
    expect(r, st == lazy_ring_ensure_status::OK, "model B's ensure returned " + name(st) + ", want ok");
    expect(r, role_bytes(r) - role0 == k_need_8k,
           "after B's ensure the ledger shows " + num(role_bytes(r) - role0) + " ring bytes, want one ring of " +
               num(k_need_8k) + " B: the parked ring outlived B's take");
    expect(r, registry().parked_at_last_take == 0,
           "B's take found " + num(registry().parked_at_last_take) + " parked node(s), want 0");
    expect(r, b.early_ring == mapped(k_need_8k) && b.late_ring == mapped(k_need_8k),
           "model B planned ring=" + num(b.early_ring) + " early and ring=" + num(b.late_ring) + " late, want " +
               num(mapped(k_need_8k)) + " on both");
    unload(b);
    return settle(r, role0);
}

// --- the failed load's end ---

// What the plan-point hook read while model B's planner held it: the registry and the published ring accessor.
struct plan_view {
    bool     entered  = false;
    bool     has_ring = false;
    uint64_t pools    = 0;
    uint64_t parked   = 0;
};

void view_plan(void * ctx) {
    plan_view *               v     = static_cast<plan_view *>(ctx);
    uint64_t                  bytes = 0;
    const lazy_registry_state s     = registry();
    v->entered                      = true;
    v->has_ring                     = lazy_block_ring_bytes(&bytes);
    v->pools                        = s.live_pools;
    v->parked                       = s.parked_nodes;
}

// Stages `inv` early under the plan-point hook and ends the load failed, so the plan is made and nothing is kept.
plan_view stage_viewed(arm_result & r, const ggml_sycl_tensor_inventory * inv, const std::string & who) {
    plan_view          view;
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, who + "'s load transaction did not begin");
        return view;
    }
    ggml_sycl_test_lazy_plan_hold(view_plan, &view);
    const ggml_sycl_lifecycle_result rc = stage(inv, true);
    ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    if (rc != GGML_SYCL_LIFECYCLE_OK) {
        stage_failed(r, who + "'s early stage", rc);
    }
    if (!view.entered) {
        void_arm(r, who + "'s planner never reached the plan point");
    }
    return view;
}

// The failed load's reap (§3.3 point (iii)): model A stages early and late with a STORAGE table and its ensure makes
// the ring; A's load then ends failed, which aborts the candidate and drops its lease, the pool's last, parking the
// pool. The guard of load_end reaps it before the call returns: no pool, live or parked, the ring's LAZY_STAGING bytes
// back to the pre-ensure figure and no ring through the accessor. Model B's planner, on the pool its own take builds,
// then sees no ring and nothing parked.
// Engagement: A's stages return OK and its ensure makes the ring.
arm_result arm_failed_load() {
    arm_result         r;
    lazy_inventory     inv_a("blk.1.failed_a.weight", k_ne0_8k, k_huge_table);
    lazy_inventory     inv_b("blk.1.failed_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t     role0 = role_bytes(r);
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "model A's load transaction did not begin");
        return settle(r, role0);
    }
    size_t late = SIZE_MAX;
    for (int pass = 0; pass < 2; ++pass) {
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv_a.inv(), pass == 0);
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, std::string("model A's ") + (pass == 0 ? "early" : "late") + " stage", rc);
            (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
            return settle(r, role0);
        }
        late = pass_ring(r, "model A's stage");
    }
    const lazy_ring_ensure_status st = block_ring_ensure(k_need_8k, k_span_8k, 0, late, txn.id);
    if (st != lazy_ring_ensure_status::OK) {
        void_arm(r, "model A's ensure returned " + name(st) + ", want ok");
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
        return settle(r, role0);
    }
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    const lazy_registry_state s = registry();
    expect(r, registry_is(s, 0, 0, 0),
           "after A's failed load_end: " + describe(s) + ", want 0, 0, 0: its guard reaps the ring the abort parked");
    expect(r, role_bytes(r) == role0 && live_ring() == 0,
           "after A's failed load_end the ring's LAZY_STAGING bytes are " + num(role_bytes(r)) +
               " and the accessor reads " + num(live_ring()) + ", want " + num(role0) + " and none");
    const plan_view view = stage_viewed(r, inv_b.inv(), "model B");
    expect(r, view.entered && !view.has_ring && view.parked == 0,
           "model B's planner saw " + std::string(view.has_ring ? "a ring" : "no ring") + " and " + num(view.parked) +
               " parked pool(s), want no ring and none");
    return settle(r, role0);
}

// --- the model swap ---

// The teardown reap (§3.3 point (ii)): model A, the q2k inventory with one STORAGE table, is loaded and its ensure
// makes the ring; A's unload erases its record, whose lease is the pool's last, and the guard declared before the
// unload's try reaps after the erase and outside that try. So A's unload returns with no pool, live or parked, and the
// ring's LAZY_STAGING bytes back, and the lease was dropped with no witnessed lock held: no "[HOST-PLAN-BUG] ring lease
// dropped with" line and no count. Model B, with no STORAGE table, then plans with no ring live, and model W, whose
// table's span is twice A's, plans normally: no decline line, and ring= its own mapped need.
// At rest the pool counts one lease per record (§3.3), so before the unload A's record holds the only one.
// Engagement: A loads and ensures, B and W stage.
// Mutant: snapshot-carries-lease, under which the published snapshot holds a second lease on the pool, so the census
// before the unload counts two. The unload alone need not show it: another holder of the snapshot, the MoE workspace
// registry, can drop the last copy outside every witnessed lock and before the teardown reap, and the ring goes back.
arm_result arm_swap() {
    arm_result       r;
    lazy_inventory   inv_a("blk.1.swap_a.weight", k_ne0_8k, k_huge_table);
    lazy_inventory   inv_w("blk.1.swap_w.weight", k_ne0_16k, k_huge_table);
    loader_inventory dense;
    const uint64_t   role0 = role_bytes(r);
    model_load       a     = load_model(r, inv_a, "model A");
    if (!a.ok) {
        return settle(r, role0);
    }
    const lazy_ring_ensure_status st = ensure_for(a);
    if (st != lazy_ring_ensure_status::OK) {
        void_arm(r, "model A's ensure returned " + name(st) + ", want ok");
        unload(a);
        return settle(r, role0);
    }
    const lazy_registry_state s0 = registry();
    expect(r, registry_is(s0, 1, 1, 0),
           "with A loaded and its ring made: " + describe(s0) + ", want 1, 1, 0: one lease, A's record's");
    log_clear();
    const ggml_sycl_lifecycle_result rc = ggml_backend_sycl_model_unloaded_token(a.token);
    const lazy_registry_state        s1 = registry();
    expect(r, rc == GGML_SYCL_LIFECYCLE_OK, "model A's unload returned " + num(rc) + ", want 0 (OK)");
    expect(r, registry_is(s1, 0, 0, 0),
           "after A's unload: " + describe(s1) + ", want 0, 0, 0: A's teardown reaps the ring its erase parked");
    expect(r, log_count_with(k_lease_locked) == 0 && s1.lease_drops_locked == s0.lease_drops_locked,
           "A's ring lease was dropped with a witnessed lock held: " + num(log_count_with(k_lease_locked)) +
               " line(s) '" + first_log_line_with(k_lease_locked) + "', counted " +
               num(s1.lease_drops_locked - s0.lease_drops_locked) + ", want 0 and 0");
    expect(r, role_bytes(r) == role0,
           "after A's unload the ring's LAZY_STAGING bytes are " + num(role_bytes(r)) + ", want " + num(role0));
    const plan_view view = stage_viewed(r, &dense.inv, "model B");
    expect(r, view.entered && !view.has_ring && view.pools == 0 && view.parked == 0,
           "model B's planner saw " + std::string(view.has_ring ? "a ring" : "no ring") + ", " + num(view.pools) +
               " live pool(s) and " + num(view.parked) + " parked, want no ring, 0 and 0");
    model_load w = load_model(r, inv_w, "model W");
    if (w.ok) {
        expect(r, w.early_ring == mapped(k_need_16k) && log_count_with("[LAZY-PLAN] STORAGE declined") == 0,
               "model W planned ring=" + num(w.early_ring) + " with " +
                   num(log_count_with("[LAZY-PLAN] STORAGE declined")) + " decline line(s), want " +
                   num(mapped(k_need_16k)) + " and 0");
        unload(w);
    }
    return settle(r, role0);
}

// --- the drop and take race ---

// The drop-first hook (§3.3's race, drop first): it runs on the stage's thread between the entry reap and the
// provisional take, lets the dropper drop the stand-in read's lease, the pool's last, and waits for the pool to park.
struct drop_first_hook {
    std::atomic<bool> entered{ false };
    std::atomic<bool> go{ false };
    std::atomic<bool> dropped{ false };
    std::atomic<bool> take_returned{ false };
    bool              parked_seen = false;
};

void drop_first_hold(void * ctx) {
    drop_first_hook * h = static_cast<drop_first_hook *>(ctx);
    h->entered          = true;
    h->go               = true;
    h->parked_seen      = wait_bounded([h] { return h->dropped.load() && registry().parked_nodes == 1; });
}

// Drop first (§3.3): with the stand-in read's lease the pool's only one, the hook between model B's entry reap and its
// provisional take has a test thread drop that lease, which parks the pool. The take then finds the slot empty, the
// node still parked, and builds a new empty pool, so B's plan charges ring= the mapped need. The dropper's own FULL
// reap runs only after the take returned, so it cannot drain the node first.
// Engagement, VOID counting as failed: the hook ran, and the node was parked when the take ran (its snapshot reads 1).
// Mutant: take-ignores-dying, under which the take joins the parked pool, so B's plan charges ring= 0.
arm_result arm_drop_first() {
    arm_result     r;
    lazy_inventory inv_b("blk.1.drop_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    ring_lease     read  = read_lease_on_live_ring(r);
    if (r.result != outcome::PASS) {
        read.reset();
        return settle(r, role0);
    }
    const uint64_t  generation_a = registry().live_generation;
    drop_first_hook hook;
    std::thread     dropper([&hook, &read] {
        if (!wait_bounded([&hook] { return hook.go.load(); })) {
            return;
        }
        read.reset();
        hook.dropped = true;
        (void) wait_bounded([&hook] { return hook.take_returned.load(); });
        ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    });
    ggml_sycl_test_lazy_stage_take_hold(drop_first_hold, &hook);
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "model B's load transaction did not begin");
    } else {
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv_b.inv(), true);
        hook.take_returned                  = true;
        ggml_sycl_test_lazy_stage_take_hold(nullptr, nullptr);
        const std::vector<size_t> rings = plan_rings();
        const size_t              ring  = rings.empty() ? SIZE_MAX : rings.front();
        const lazy_registry_state s     = registry();
        if (!hook.entered || !hook.parked_seen || s.parked_at_last_take != 1) {
            void_arm(r, std::string("the stage ") +
                            (hook.entered ? "held between its entry reap and its take saw the pool parked: " +
                                                std::string(hook.parked_seen ? "yes" : "no") + ", and its take found " +
                                                num(s.parked_at_last_take) + " parked node(s), want 1" :
                                            std::string("did not reach the hold between its entry reap and its take")));
        }
        expect(r, ring == mapped(k_need_8k),
               "model B's early plan charged ring=" + num(ring) + ", want " + num(mapped(k_need_8k)) +
                   ": the take joined the parked pool");
        expect(r, s.live_generation != generation_a,
               "model B's pool is generation " + num(s.live_generation) + ", A's, want a new one");
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, "model B's early stage", rc);
        }
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    }
    hook.go            = true;
    hook.take_returned = true;
    ggml_sycl_test_lazy_stage_take_hold(nullptr, nullptr);
    dropper.join();
    read.reset();
    return settle(r, role0);
}

// The take-first hooks: the drop's hook holds the dropper before its critical section; the plan-point hook, on the
// stage's thread after its take, releases the dropper and waits for its drop to return.
struct take_first_hook {
    bounded_gate      drop_gate;
    std::atomic<bool> dropped{ false };
    bool              entered      = false;
    bool              resumed      = false;
    uint64_t          pools_after  = 0;
    uint64_t          parked_after = 0;
};

void take_first_plan(void * ctx) {
    take_first_hook * h = static_cast<take_first_hook *>(ctx);
    h->entered          = true;
    h->drop_gate.release();
    h->resumed                  = wait_bounded([h] { return h->dropped.load(); });
    const lazy_registry_state s = registry();
    h->pools_after              = s.live_pools;
    h->parked_after             = s.parked_nodes;
}

// Take first (§3.3): the drop's hook holds a test thread's drop of the stand-in read's lease before its critical
// section; model B's stage takes, joining the live pool; inside B's planner the dropper resumes, its decrement does not
// reach zero and nothing parks. B plans ring= 0, covered, and the ring's bytes through the accessor are unchanged.
// Engagement, VOID counting as failed: the drop's hook held the dropper within 5 s, and the dropper resumed after the
// take returned, inside B's planner.
arm_result arm_take_first() {
    arm_result     r;
    lazy_inventory inv_b("blk.1.take_b.weight", k_ne0_8k, k_huge_table);
    const uint64_t role0 = role_bytes(r);
    ring_lease     read  = read_lease_on_live_ring(r);
    if (r.result != outcome::PASS) {
        read.reset();
        return settle(r, role0);
    }
    const uint64_t  ring0 = live_ring();
    take_first_hook hook;
    ggml_sycl_test_lazy_drop_hold([](void * ctx) { static_cast<bounded_gate *>(ctx)->hold(); }, &hook.drop_gate);
    std::thread dropper([&hook, &read] {
        read.reset();
        hook.dropped = true;
    });
    if (!hook.drop_gate.wait_holders(1)) {
        ggml_sycl_test_lazy_drop_hold(nullptr, nullptr);
        hook.drop_gate.release();
        dropper.join();
        void_arm(r, "the drop did not reach its hold within " + num(k_bound.count()) + " s");
        return settle(r, role0);
    }
    ggml_sycl_test_lazy_plan_hold(take_first_plan, &hook);
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "model B's load transaction did not begin");
    } else {
        log_clear();
        const ggml_sycl_lifecycle_result rc = stage(inv_b.inv(), true);
        ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
        const std::vector<size_t> rings = plan_rings();
        const size_t              ring  = rings.empty() ? SIZE_MAX : rings.front();
        if (!hook.entered || !hook.resumed) {
            void_arm(r, hook.entered ? "the dropper did not resume within " + num(k_bound.count()) +
                                           " s of its release inside B's planner" :
                                       std::string("B's planner never reached the plan point"));
        }
        expect(r, hook.pools_after == 1 && hook.parked_after == 0,
               "after the dropper resumed: " + num(hook.pools_after) + " live pool(s) and " + num(hook.parked_after) +
                   " parked, want 1 and 0: its decrement reached zero");
        expect(r, ring == 0, "model B's early plan charged ring=" + num(ring) + ", want 0: the ring was live");
        expect(r, live_ring() == ring0,
               "the ring through the accessor reads " + num(live_ring()) + " B, want " + num(ring0) + " B");
        if (rc != GGML_SYCL_LIFECYCLE_OK) {
            stage_failed(r, "model B's early stage", rc);
        }
        (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);
    }
    ggml_sycl_test_lazy_plan_hold(nullptr, nullptr);
    hook.drop_gate.release();
    dropper.join();
    return settle(r, role0);
}

// --- the children ---

// A child's way out: its line, then _Exit, so nothing is torn down after a shutdown or under a planted mutant.
[[noreturn]] void child_exit(const char * tag, const std::string & line, int code) {
    std::fprintf(stderr, "%s %s\n", tag, line.c_str());
    std::fflush(stderr);
    std::_Exit(code);
}

// A child's backends, its multi-device check and model A loaded and ensured; exits 77 naming why it cannot engage.
struct child_model {
    std::vector<ggml_backend_t> backends;
    lazy_inventory              inv{ "blk.1.child_a.weight", k_ne0_8k, k_huge_table };
    model_load                  a;
    uint64_t                    role0 = 0;
};

void child_setup(const char * tag, child_model & m) {
    if (!ggml_backend_sycl_moe_multi_gpu_requested()) {
        child_exit(tag, "void: the planner caller takes its single-device path", SYCL_TEST_SKIP);
    }
    // A child leaves main before main installs the capture, and load_model reads each stage's ring from the captured
    // [HOST-PLAN] lazy line, so the child installs its own. capture_log echoes every line to stderr, where the parent
    // scores the child.
    ggml_log_set(capture_log, nullptr);
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        m.backends.push_back(ggml_backend_sycl_init(d));
        if (m.backends.back() == nullptr) {
            child_exit(tag, "void: backend " + std::to_string(d) + " did not initialise", SYCL_TEST_SKIP);
        }
    }
    // The ring's LAZY_STAGING reading needs device 0's cache, which neither the backends' init nor the ring's HOST_CPU
    // allocation builds; main builds it the same way before any arm.
    if (get_unified_cache_for_device(0) == nullptr) {
        child_exit(tag, "void: device 0's cache was not created, so every LAZY_STAGING reading would be VOID",
                   SYCL_TEST_SKIP);
    }
    arm_result r;
    m.role0 = role_bytes(r);
    m.a     = load_model(r, m.inv, "model A");
    if (!m.a.ok || r.result != outcome::PASS) {
        child_exit(tag, "void: " + r.why, SYCL_TEST_SKIP);
    }
    const lazy_ring_ensure_status st = ensure_for(m.a);
    if (st != lazy_ring_ensure_status::OK) {
        child_exit(tag, "void: model A's ensure returned " + name(st), SYCL_TEST_SKIP);
    }
}

std::string registry_line() {
    const lazy_registry_state s = registry();
    return "pools=" + num(s.live_pools) + " leases=" + num(s.live_leases) + " parked=" + num(s.parked_nodes);
}

// The backend shutdown, whose throw the child reports instead of ending with it.
std::string shut_down() {
    try {
        ggml_backend_sycl_shutdown();
        return "none";
    } catch (const std::exception & e) {
        return e.what();
    } catch (...) {
        return "unknown";
    }
}

int throw_child() {
    child_model m;
    child_setup("TEARDOWN-CHILD", m);
    ggml_sycl_test_lazy_reap_throw_next();
    const ggml_sycl_lifecycle_result rc = ggml_backend_sycl_model_unloaded_token(m.a.token);
    const lazy_registry_state        s  = registry();
    std::fprintf(stderr, "TEARDOWN-CHILD unloaded rc=%d failed=%llu parked=%llu\n", (int) rc,
                 (unsigned long long) s.reaps_failed, (unsigned long long) s.parked_nodes);
    const ggml_sycl_lifecycle_result again = ggml_backend_sycl_model_unloaded_token(m.a.token);
    std::fprintf(stderr, "TEARDOWN-CHILD again rc=%d\n", (int) again);
    std::fflush(stderr);
    ggml_sycl_lazy_ring_reap(lazy_reap_scope::FULL);
    child_exit("TEARDOWN-CHILD", "after the next reap parked=" + num(registry().parked_nodes), 0);
}

// The shutdown children: `read` holds a read in flight across the shutdown, released by the shutdown's own hook once
// it is about to wait for the picked reads; `freed` unloads A and lets the read finish first, waiting for the ring's
// LAZY_STAGING bytes to come back, and then shuts down.
void release_gate(void * ctx) {
    static_cast<gate *>(ctx)->release();
}

int shutdown_child(const char * which) {
    child_model m;
    child_setup("SHUTDOWN-CHILD", m);
    std::unique_ptr<held_read> read(new held_read());
    arm_result                 r;
    if (!read->start(r)) {
        child_exit("SHUTDOWN-CHILD", "void: " + r.why, SYCL_TEST_SKIP);
    }
    const bool freed = std::strcmp(which, "freed") == 0;
    unload(m.a);
    if (freed) {
        read->g.release();
        const bool back = wait_bounded([&m] {
            arm_result ignored;
            return role_bytes(ignored) == m.role0;
        });
        arm_result ignored;
        std::fprintf(stderr, "SHUTDOWN-CHILD freed back=%d role=%llu role0=%llu\n", back ? 1 : 0,
                     (unsigned long long) role_bytes(ignored), (unsigned long long) m.role0);
    } else {
        ggml_sycl_test_lazy_shutdown_hold(release_gate, &read->g);
    }
    const std::string threw     = shut_down();
    const bool        delivered = read->ticket.outstanding.load() == 0;
    const std::string reg       = registry_line();
    const bool        take      = lazy_io_pool_take().held();
    const bool        clean     = unified_cache_shutdown_state_clean();
    child_exit("SHUTDOWN-CHILD",
               "returned " + reg + " threw=" + threw + " delivered=" + (delivered ? "1" : "0") +
                   " take=" + (take ? "1" : "0") + " clean=" + (clean ? "1" : "0"),
               0);
}

int leaked_child() {
    child_model m;
    child_setup("LEAKED-CHILD", m);
    std::fprintf(stderr, "LEAKED-CHILD loaded model=%llu load=%llu\n", (unsigned long long) m.a.token.model_id,
                 (unsigned long long) m.a.token.load_txn_id);
    const std::string threw   = shut_down();
    const size_t      records = lifecycle_plan_lease_count_for_test();
    std::fprintf(stderr, "LEAKED-CHILD returned threw=%s records=%zu\n", threw.c_str(), records);
    lifecycle_plan_clear_for_test();
    child_exit("LEAKED-CHILD", "cleared drops_after_shut_down=" + num(registry().drops_after_shut_down), 0);
}

// --- the child-scored arms ---

child_run g_throw_run;
child_run g_shutdown_read_run;
child_run g_shutdown_freed_run;
child_run g_leaked_run;

child_run spawn(std::vector<std::string> args) {
    if (g_child_mutant != nullptr) {
        args.emplace_back(g_child_mutant);
    }
    return run_child_strings(args, child_environment({ "GGML_NO_BACKTRACE=1" }), k_child_bound);
}

// A child that did not end, or exited 77 naming why it could not engage, VOIDs the arm.
bool child_engaged(arm_result & r, const child_run & run, const char * tag) {
    if (!run.ran) {
        void_arm(r, std::string("the child did not end within ") + num(k_child_bound.count() / 1000) + " s");
        return false;
    }
    const std::string v = first_line_with(run.err, std::string(tag) + " void:");
    if (v != "(none)") {
        void_arm(r, v);
        return false;
    }
    return true;
}

bool exited_zero(const child_run & run) {
    return WIFEXITED(run.status) && WEXITSTATUS(run.status) == 0;
}

// The throwing teardown reap (§3.3: a failure to reset is logged [HOST-PLAN-BUG] and never thrown): with the next FULL
// reap's first step made to throw std::system_error, model A's unload returns OK, its reap logs one "ring reap failed"
// line and leaves A's pool parked, and the ticket stays finalized: a second unload of the token reports it dead. The
// next reap frees the pool.
// Engagement: the child's backends initialise and A loads and ensures.
// Mutant: reap-inside-try, under which the teardown reap runs inside the try that re-finalizes the ticket, so the
// throw reaches its catch and the unload returns EFFECT_FAILED.
arm_result arm_throw() {
    arm_result r;
    if (!child_engaged(r, g_throw_run, "TEARDOWN-CHILD")) {
        return r;
    }
    const std::string & err      = g_throw_run.err;
    const std::string   unloaded = first_line_with(err, "TEARDOWN-CHILD unloaded");
    expect(r, exited_zero(g_throw_run),
           "the teardown's throwing reap ended the child with " + status_text(g_throw_run.status) + " after '" +
               unloaded + "', want exit 0");
    const std::string want = "TEARDOWN-CHILD unloaded rc=0 failed=1 parked=1";
    expect(r, unloaded == want, "the teardown printed '" + unloaded + "', want '" + want + "'");
    expect(r, text_count_with(err, k_reap_failed) == 1,
           "the teardown printed " + num(text_count_with(err, "[HOST-PLAN-BUG] ring reap failed")) +
               " reap-failed line(s), want 1 '" + k_reap_failed + "'");
    const std::string again = "TEARDOWN-CHILD again rc=" + std::to_string((int) GGML_SYCL_LIFECYCLE_OK_ALREADY_DEAD);
    expect(r, text_count_with(err, again.c_str()) == 1,
           "the second unload printed '" + first_line_with(err, "TEARDOWN-CHILD again") + "', want '" + again + "'");
    expect(r, text_count_with(err, "TEARDOWN-CHILD after the next reap parked=0") == 1,
           "the next reap printed '" + first_line_with(err, "TEARDOWN-CHILD after the next reap") + "', want parked=0");
    return r;
}

// The shutdown drain (§3.3 point (v)) with a read in flight: A is unloaded while a read holds its lease on a worker,
// and the shutdown's hook releases the read once the queued reads are cancelled. The shutdown returns with the read
// delivered, no pool, live or parked, the registry shut down so a take returns no lease, and a clean cache census.
// Engagement: the child's read is held on a worker.
arm_result score_shutdown(const child_run & run, const char * what) {
    arm_result r;
    if (!child_engaged(r, run, "SHUTDOWN-CHILD")) {
        return r;
    }
    const std::string got = first_line_with(run.err, "SHUTDOWN-CHILD returned");
    expect(r, exited_zero(run) && got != "(none)",
           std::string("the shutdown ") + what + " ended its child with " + status_text(run.status) + ", want exit 0");
    expect(r, got.find(" pools=0 leases=0 parked=0 ") != std::string::npos,
           std::string("after the shutdown ") + what + ": '" + got + "', want pools=0 leases=0 parked=0");
    expect(r, got.find(" threw=none ") != std::string::npos,
           std::string("the shutdown ") + what + " threw: '" + got + "'");
    expect(r, got.find(" delivered=1 take=0 clean=1") != std::string::npos,
           std::string("after the shutdown ") + what + ": '" + got + "', want delivered=1 take=0 clean=1");
    return r;
}

arm_result arm_shutdown_read() {
    return score_shutdown(g_shutdown_read_run, "with a read in flight");
}

// The shutdown drain with the last table and read freed first: A's unload and the read's delivery give the ring's
// LAZY_STAGING bytes back to the pre-ensure figure through the reader-thread reap, and the shutdown that follows leaves
// no pool and a clean census.
arm_result arm_shutdown_freed() {
    arm_result r = score_shutdown(g_shutdown_freed_run, "with the last table and read freed");
    if (r.result == outcome::VOID) {
        return r;
    }
    const std::string got = first_line_with(g_shutdown_freed_run.err, "SHUTDOWN-CHILD freed");
    arm_result        freed;
    expect(freed, got.find("back=1 ") != std::string::npos,
           "the freed ring's LAZY_STAGING bytes did not return within " + num(k_bound.count()) + " s: '" + got + "'");
    if (freed.result != outcome::PASS) {
        return freed;
    }
    return r;
}

// The leaked-record arm (§3.3's drain): model A is loaded and never unloaded, so its published record holds the
// pool's lease through the shutdown. The drain moves that lease out, logs one "[HOST-PLAN-BUG] leaked ring lease:
// model=<m> load=<t>" line naming A, and drops it with no lock held: after the shutdown no record holds a lease, and
// once the records are cleared, as static destruction would, no held lease's destructor runs after shut-down.
// Engagement: the child's backends initialise and A loads and ensures.
// Mutant: skip-lease-drain, under which the drain moves no lease, so clearing the records destroys a held lease after
// shut-down.
arm_result arm_leaked() {
    arm_result r;
    if (!child_engaged(r, g_leaked_run, "LEAKED-CHILD")) {
        return r;
    }
    const std::string & err    = g_leaked_run.err;
    const std::string   loaded = first_line_with(err, "LEAKED-CHILD loaded ");
    const std::string   ids    = loaded == "(none)" ? std::string() : loaded.substr(loaded.find("model="));
    expect(r, exited_zero(g_leaked_run) && text_count_with(err, "LEAKED-CHILD cleared") == 1,
           "the leaked-record child ended with " + status_text(g_leaked_run.status) + ", want exit 0");
    expect(r, text_count_with(err, "LEAKED-CHILD cleared drops_after_shut_down=0") == 1,
           "after the records were cleared: '" + first_line_with(err, "LEAKED-CHILD cleared") +
               "', want drops_after_shut_down=0");
    expect(r, text_count_with(err, "LEAKED-CHILD returned threw=none records=0") == 1,
           "after the shutdown: '" + first_line_with(err, "LEAKED-CHILD returned") + "', want threw=none records=0");
    const std::string want = ids.empty() ? std::string(k_leaked) : std::string(k_leaked) + ids.substr(6);
    expect(
        r, !ids.empty() && text_count_with(err, want.c_str()) == 1 && text_count_with(err, k_leaked) == 1,
        "the shutdown printed " + num(text_count_with(err, k_leaked)) + " leaked-lease line(s), want 1 '" + want + "'");
    return r;
}

// --- the mutants ---

struct mutant_entry {
    const char *     text;
    lazy_ring_mutant mutant;
    const char *     arm;
};

const mutant_entry k_mutants[] = {
    { "no-entry-reap",          lazy_ring_mutant::NO_ENTRY_REAP,          k_arm_entry      },
    { "snapshot-carries-lease", lazy_ring_mutant::SNAPSHOT_CARRIES_LEASE, k_arm_swap       },
    { "take-ignores-dying",     lazy_ring_mutant::TAKE_IGNORES_DYING,     k_arm_drop_first },
    { "reap-inside-try",        lazy_ring_mutant::REAP_INSIDE_TRY,        k_arm_throw      },
    { "skip-lease-drain",       lazy_ring_mutant::SKIP_LEASE_DRAIN,       k_arm_leaked     },
};

const mutant_entry * mutant_named(const char * text) {
    for (const mutant_entry & e : k_mutants) {
        if (std::strcmp(e.text, text) == 0) {
            return &e;
        }
    }
    return nullptr;
}

// A child's mutant, planted before it does anything; an unknown name ends it as a failure.
void plant_child_mutant(const char * tag, const char * text) {
    if (text == nullptr) {
        return;
    }
    const mutant_entry * e = mutant_named(text);
    if (e == nullptr) {
        child_exit(tag, std::string("unknown mutant ") + text, 2);
    }
    ggml_sycl_test_lazy_ring_mutant_select(e->mutant);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--throw-child") == 0) {
        plant_child_mutant("TEARDOWN-CHILD", argc >= 3 ? argv[2] : nullptr);
        return throw_child();
    }
    if (argc >= 3 && std::strcmp(argv[1], "--shutdown-child") == 0) {
        plant_child_mutant("SHUTDOWN-CHILD", argc >= 4 ? argv[3] : nullptr);
        return shutdown_child(argv[2]);
    }
    if (argc >= 2 && std::strcmp(argv[1], "--leaked-child") == 0) {
        plant_child_mutant("LEAKED-CHILD", argc >= 3 ? argv[2] : nullptr);
        return leaked_child();
    }
    const mutant_entry * mutant = nullptr;
    if (argc == 3 && std::strcmp(argv[1], "--mutant") == 0) {
        mutant = mutant_named(argv[2]);
        if (mutant == nullptr) {
            std::printf("FAIL: unknown mutant %s\n", argv[2]);
            return 1;
        }
    } else if (argc != 1) {
        std::printf("FAIL: unknown arguments; the option is --mutant <name>\n");
        return 1;
    }
    if (ggml_row_size(GGML_TYPE_Q8_0, k_ne0_8k) != k_row_8k || ggml_row_size(GGML_TYPE_Q8_0, k_ne0_16k) != k_row_16k) {
        std::printf("FAIL: the Q8_0 row sizes are not the geometry's 34 and 8194 B\n");
        return 1;
    }

    // The children start before this process initialises SYCL, one at a time: every one without a mutant, else the
    // one the mutant's arm scores.
    g_child_mutant    = mutant != nullptr ? mutant->text : nullptr;
    const auto wanted = [mutant](const char * arm) {
        return mutant == nullptr || std::strcmp(mutant->arm, arm) == 0;
    };
    if (wanted(k_arm_throw)) {
        g_throw_run = spawn({ k_self, "--throw-child" });
        std::fputs(g_throw_run.err.c_str(), stderr);
    }
    if (wanted(k_arm_shutdown_read)) {
        g_shutdown_read_run = spawn({ k_self, "--shutdown-child", "read" });
        std::fputs(g_shutdown_read_run.err.c_str(), stderr);
    }
    if (wanted(k_arm_shutdown_freed)) {
        g_shutdown_freed_run = spawn({ k_self, "--shutdown-child", "freed" });
        std::fputs(g_shutdown_freed_run.err.c_str(), stderr);
    }
    if (wanted(k_arm_leaked)) {
        g_leaked_run = spawn({ k_self, "--leaked-child" });
        std::fputs(g_leaked_run.err.c_str(), stderr);
    }

    if (!sycl_test_require_gpu("the lazy reap point arms")) {
        return SYCL_TEST_SKIP;
    }
    if (!ggml_backend_sycl_moe_multi_gpu_requested()) {
        std::printf(
            "SKIP: the planner caller takes its single-device path: ggml_backend_sycl_moe_multi_gpu_requested() "
            "is false\n");
        return SYCL_TEST_SKIP;
    }
    ggml_log_set(capture_log, nullptr);
    std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
    for (int d = 0; d < ggml_backend_sycl_get_device_count(); ++d) {
        backends.emplace_back(ggml_backend_sycl_init(d), ggml_backend_free);
        if (!backends.back()) {
            std::printf("FAIL: backend %d did not initialise\n", d);
            return 1;
        }
    }
    if (get_unified_cache_for_device(0) == nullptr) {
        std::printf("FAIL: device 0's cache was not created, so every LAZY_STAGING reading would be VOID\n");
        return 1;
    }

    const std::vector<named_arm> arms = {
        { k_arm_entry,          arm_entry          },
        { k_arm_failed_load,    arm_failed_load    },
        { k_arm_swap,           arm_swap           },
        { k_arm_drop_first,     arm_drop_first     },
        { k_arm_take_first,     arm_take_first     },
        { k_arm_throw,          arm_throw          },
        { k_arm_shutdown_read,  arm_shutdown_read  },
        { k_arm_shutdown_freed, arm_shutdown_freed },
        { k_arm_leaked,         arm_leaked         },
    };
    if (mutant == nullptr) {
        return run_arms(arms, "lazy reap point arms");
    }
    ggml_sycl_test_lazy_ring_mutant_select(mutant->mutant);
    std::vector<named_arm> target;
    for (const named_arm & arm : arms) {
        if (std::strcmp(arm.name, mutant->arm) == 0) {
            target.push_back(arm);
        }
    }
    return run_arms(target, std::string("lazy reap point arms (--mutant ") + mutant->text + ")");
}
