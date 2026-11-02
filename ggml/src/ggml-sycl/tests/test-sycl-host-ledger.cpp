//
// The process-global host ledger, arms (a)-(i) of th32 §11 T2: the lock-free charge protocol, the yield and its
// unregister handshake, the credit, and the saturating budget function.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// These arms touch only the ledger functions, a fake yielder and the budget function. Two more legs score the plan's
// host bytes through the two pure functions behind them, the commit terms and the zone commit check, on hand-built
// plans. Two score the re-record of a plan's terms by the re-plans that grow a term after the record, the KV refresh
// and the runtime MMID re-plan, on hand-built plans; the MMID re-plan reads its owner bound from a mocked device table.
// One runs the host sample in child processes to score its lock check. None constructs a cache or a pool and none
// initialises SYCL, so the binary needs no device. Arms (j)-(o), the over-budget fixture and the plan-against-committed
// fixture run a planner, a pool or the HOST_CPU tier and are in test-sycl-host-ledger-device.cpp, which the lead runs.
//
// Every arm names its engagement check, the observation that proves it reached the mechanism it scores, and the
// mutant it kills. An arm whose engagement check fails is VOID, and a VOID arm fails the binary like a FAIL. Every arm
// runs, so one failure does not hide the rest, except a charge that never returns: the binary stops at that one.

#include "../common.hpp"
#include "../ggml-sycl-test.hpp"
#include "../kv-region-registry.hpp"
#include "../unified-cache.hpp"
#include "host-ledger-commit-oracle.hpp"

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
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

namespace {

constexpr size_t GiB = 1024ull * 1024ull * 1024ull;

// The bound on any wait in this file. A charge that has not returned by then is an unbounded loop (arm (e)'s mutant)
// or a lost release, and the arm is VOID.
constexpr auto k_bound = std::chrono::seconds(5);

enum class outcome { PASS, FAIL, VOID };

struct arm_result {
    outcome     result = outcome::PASS;
    std::string why;
};

// The first FAIL or VOID wins; a later check does not overwrite the reason.
void fail(arm_result & r, const std::string & why) {
    if (r.result == outcome::PASS) {
        r.result = outcome::FAIL;
        r.why    = why;
    }
}

void void_arm(arm_result & r, const std::string & why) {
    if (r.result == outcome::PASS) {
        r.result = outcome::VOID;
        r.why    = why;
    }
}

void expect(arm_result & r, bool condition, const std::string & why) {
    if (!condition) {
        fail(r, why);
    }
}

std::string num(size_t v) {
    return std::to_string(static_cast<unsigned long long>(v));
}

// --- the log lines the arms read ---

std::mutex               g_log_mutex;
std::vector<std::string> g_log_lines;

void capture_log(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text == nullptr) {
        return;
    }
    fputs(text, stderr);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_lines.emplace_back(text);
}

void log_clear() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_lines.clear();
}

// The last captured line that contains every needle, or "" when none does.
std::string log_line_with(const char * a, const char * b = nullptr, const char * c = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    for (size_t i = g_log_lines.size(); i-- > 0;) {
        const std::string & line = g_log_lines[i];
        if (line.find(a) == std::string::npos) {
            continue;
        }
        if (b && line.find(b) == std::string::npos) {
            continue;
        }
        if (c && line.find(c) == std::string::npos) {
            continue;
        }
        return line;
    }
    return "";
}

size_t log_count_with(const char * a) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    size_t                      n = 0;
    for (const std::string & line : g_log_lines) {
        n += line.find(a) != std::string::npos ? 1 : 0;
    }
    return n;
}

// The value printed after " <name>=" on the line, or SIZE_MAX when the name is absent.
size_t printed(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t      at  = line.find(key);
    if (at == std::string::npos) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(std::strtoull(line.c_str() + at + key.size(), nullptr, 10));
}

// --- ledger state the arms share ---

void clear_hooks() {
    for (int32_t p = 0; p < GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT; ++p) {
        (void) ggml_sycl_test_host_ledger_hook(p, nullptr);
    }
}

// Every arm starts from committed 0, no hook and no yieldable chunk, and puts back what it charged.
void ledger_settle() {
    clear_hooks();
    const size_t committed = host_ledger_committed();
    if (committed != 0) {
        host_ledger_credit(committed);
    }
    ggml_sycl_test_host_ledger_store_budget(0);
    log_clear();
}

host_ledger_charge_result charge(size_t bytes, bool may_yield = false, uint64_t load_txn_id = 0) {
    host_ledger_charge_request req{};
    req.bytes       = bytes;
    req.may_yield   = may_yield;
    req.load_txn_id = load_txn_id;
    return host_ledger_try_charge(req);
}

// n yieldable chunks of `bytes` each, for the arm's lifetime.
struct yieldable_chunks {
    size_t n;
    size_t bytes;

    yieldable_chunks(size_t n_, size_t bytes_) : n(n_), bytes(bytes_) {
        for (size_t i = 0; i < n; ++i) {
            host_ledger_yieldable_add(bytes);
        }
    }

    ~yieldable_chunks() {
        for (size_t i = 0; i < n; ++i) {
            host_ledger_yieldable_sub(bytes);
        }
    }
};

// The fake yielder: counts its calls, frees (and credits) a fixed amount per call, and reports a sentinel set after
// its unregister if a call ever sees it.
struct fake_yielder {
    std::atomic<int>      calls{ 0 };
    size_t                free_per_call = 0;
    std::atomic<bool>     destroyed{ false };
    std::atomic<bool>     saw_destroyed{ false };
    std::atomic<uint64_t> last_txn{ 0 };
    // arm (f): the callback's own charge, made from inside the callback
    size_t                inner_ask = 0;
    std::atomic<bool>     inner_tried{ false };
    std::atomic<bool>     inner_charged{ false };
    std::atomic<int>      depth{ 0 };
    int                   slot = -1;

    static size_t yield(void * ctx, size_t deficit, uint64_t load_txn_id) noexcept {
        (void) deficit;
        fake_yielder * self = static_cast<fake_yielder *>(ctx);
        self->calls.fetch_add(1);
        self->last_txn.store(load_txn_id);
        if (self->destroyed.load()) {
            self->saw_destroyed.store(true);
            return 0;
        }
        const int depth = self->depth.fetch_add(1);
        if (self->inner_ask != 0 && depth == 0) {
            // A charge the callback makes asks to yield; it must refuse at once, since a yield never recurses.
            self->inner_tried.store(true);
            const host_ledger_charge_result inner = charge(self->inner_ask, true);
            self->inner_charged.store(inner.charged);
            if (inner.charged) {
                host_ledger_credit(inner.ask);
            }
        }
        self->depth.fetch_sub(1);
        if (depth != 0) {
            return 0;  // a recursive call (the mutant's) frees nothing, so the recursion ends here
        }
        if (self->free_per_call != 0) {
            host_ledger_credit(self->free_per_call);
        }
        return self->free_per_call;
    }

    bool attach() {
        slot = host_ledger_register_yielder(&fake_yielder::yield, this);
        return slot >= 0;
    }

    void detach() {
        if (slot >= 0) {
            host_ledger_unregister_yielder(slot);
            slot = -1;
        }
    }

    ~fake_yielder() { detach(); }
};

// Waits up to k_bound for `flag`; false on timeout.
bool wait_for(const std::atomic<bool> & flag) {
    const auto deadline = std::chrono::steady_clock::now() + k_bound;
    while (!flag.load()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Runs a charge on its own thread. One that has not returned within k_bound is an unbounded loop: the arm is VOID, and
// since the spinning thread still calls the arm's yielder, the binary stops there instead of running the later arms
// beside it.
host_ledger_charge_result bounded_charge(size_t bytes, bool may_yield, const char * what) {
    struct state {
        std::atomic<bool>         done{ false };
        host_ledger_charge_result result{};
    };

    std::shared_ptr<state> st = std::make_shared<state>();
    std::thread            t([st, bytes, may_yield] {
        st->result = charge(bytes, may_yield);
        st->done.store(true);
    });
    if (!wait_for(st->done)) {
        std::printf("  VOID: %s did not return within 5 s; the binary stops here\n", what);
        std::fflush(stdout);
        std::_Exit(1);
    }
    t.join();
    return st->result;
}

// --- the hold hook the race arms share: holds the first firing on a marked thread until released ---

struct hold_state {
    int32_t           point = -1;
    std::atomic<bool> held{ false };
    std::atomic<bool> release{ false };
    std::atomic<bool> timed_out{ false };
    std::atomic<int>  fired_elsewhere{ 0 };
};

hold_state        g_hold;
thread_local bool t_hold_this_thread = false;

int32_t hold_hook(int32_t point, int32_t arg) {
    (void) arg;
    if (point != g_hold.point) {
        return 0;
    }
    if (!t_hold_this_thread) {
        g_hold.fired_elsewhere.fetch_add(1);
        return 0;
    }
    t_hold_this_thread = false;  // only the first firing on the marked thread is held
    g_hold.held.store(true);
    if (!wait_for(g_hold.release)) {
        g_hold.timed_out.store(true);
    }
    return 0;
}

void hold_arm(int32_t point) {
    g_hold.point = point;
    g_hold.held.store(false);
    g_hold.release.store(false);
    g_hold.timed_out.store(false);
    g_hold.fired_elsewhere.store(0);
}

// --- the arms ---

// (a) Two threads race charges that together exceed the budget by one byte. The hook holds the first between its load
// and its CAS until the second has charged; exactly one succeeds and committed never passes the budget. Then a store
// that moves the budget between two charges does not let a stale committed through.
// Engagement: the hook held the first thread at CHARGE_BEFORE_CAS while the second charged.
// Mutant: a load-then-add in place of the CAS lets both through (committed = budget + 1).
arm_result arm_a() {
    arm_result r;
    ledger_settle();
    const size_t budget = 1000;
    const size_t x      = 600;
    const size_t y      = budget + 1 - x;
    ggml_sycl_test_host_ledger_store_budget(budget);
    hold_arm(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS);
    if (!ggml_sycl_test_host_ledger_hook(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS, hold_hook)) {
        void_arm(r, "the CHARGE_BEFORE_CAS hook could not be set");
        return r;
    }
    host_ledger_charge_result first{};
    std::atomic<bool>         first_done{ false };
    std::thread               t([&] {
        t_hold_this_thread = true;
        first              = charge(x);
        first_done.store(true);
    });
    if (!wait_for(g_hold.held)) {
        g_hold.release.store(true);
        t.join();
        void_arm(r, "the hook never held the first charge between its load and its CAS");
        ledger_settle();
        return r;
    }
    const host_ledger_charge_result second                 = charge(y);
    const bool                      second_done_while_held = !first_done.load() && g_hold.held.load();
    g_hold.release.store(true);
    t.join();
    if (g_hold.timed_out.load() || !second_done_while_held) {
        void_arm(r, "the second charge did not complete while the first was held");
    }
    const size_t committed = host_ledger_committed();
    expect(r, first.charged != second.charged,
           "exactly one of two charges summing to budget + 1 succeeds (first=" + num(first.charged) +
               " second=" + num(second.charged) + ")");
    expect(r, committed <= budget, "committed " + num(committed) + " never passes the budget " + num(budget));
    expect(r, committed == (first.charged ? x : 0) + (second.charged ? y : 0),
           "committed is the sum of the charges that succeeded");

    // A store between two charges moves the limit without letting a stale committed through.
    ledger_settle();
    ggml_sycl_test_host_ledger_store_budget(100);
    const host_ledger_charge_result c1 = charge(60);
    if (!c1.charged) {
        void_arm(r, "the store-between-charges case could not make its first charge");
    }
    ggml_sycl_test_host_ledger_store_budget(50);
    const host_ledger_charge_result c2 = charge(1);
    expect(r, !c2.charged, "a charge after the budget is stored below committed is refused");
    ggml_sycl_test_host_ledger_store_budget(200);
    const host_ledger_charge_result c3 = charge(1);
    expect(r, c3.charged, "a charge after the budget is stored above committed fits");
    expect(r, host_ledger_committed() == 61, "committed is 61 after the two charges that fit");
    ledger_settle();
    return r;
}

// (b) A deficit above yieldable_bytes refuses with host-budget-exceeded and never calls the yielder.
// Engagement: the yielder is registered and the shortfall's refusal line was logged.
// Mutant: the deficit pre-check removed calls the yielder once.
arm_result arm_b() {
    arm_result r;
    ledger_settle();
    fake_yielder y;
    y.free_per_call = 10;
    if (!y.attach()) {
        void_arm(r, "the fake yielder could not register");
        return r;
    }
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(90).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    yieldable_chunks          chunks(1, 5);
    host_ledger_charge_result c    = charge(20, true);
    const std::string         line = log_line_with("host-budget-exceeded");
    if (line.empty()) {
        void_arm(r, "no host-budget-exceeded line: the shortfall path did not run");
    }
    expect(r, !c.charged, "a deficit of 10 above yieldable_bytes 5 is refused");
    expect(r, y.calls.load() == 0, "the yielder is never called (calls=" + num(y.calls.load()) + ")");
    expect(r, printed(line, "deficit") == 10, "the line names deficit=10");
    expect(r, printed(line, "yieldable_bytes") == 5, "the line names yieldable_bytes=5");
    y.detach();
    ledger_settle();
    return r;
}

// (c) A shortfall that one yield round covers succeeds, with the yielder called once.
// Engagement: the charge was short before the round (the yielder was called).
// Mutant: no retry after the round refuses with the yielder called once.
arm_result arm_c() {
    arm_result r;
    ledger_settle();
    fake_yielder y;
    y.free_per_call = 10;
    if (!y.attach()) {
        void_arm(r, "the fake yielder could not register");
        return r;
    }
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(90).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    yieldable_chunks                chunks(2, 10);
    const host_ledger_charge_result c = charge(20, true);
    if (y.calls.load() == 0) {
        void_arm(r, "the yielder was never called: the charge was not short");
    }
    expect(r, c.charged, "a shortfall of 10 that one round frees fits");
    expect(r, y.calls.load() == 1, "the yielder is called once (calls=" + num(y.calls.load()) + ")");
    expect(r, c.retries == 1, "one yield round is reported (retries=" + num(c.retries) + ")");
    expect(r, host_ledger_committed() == 100, "committed is the budget after the round and the charge");
    y.detach();
    ledger_settle();
    return r;
}

// (d) A request without may_yield refuses at once with the yielder never called, and its line logs yieldable_bytes.
// The flag defaults to false on both the ledger's request and the allocator's.
// Engagement: the shortfall's refusal line was logged with yieldable chunks present.
// Mutant: may_yield defaulting to true calls the yielder.
arm_result arm_d() {
    arm_result r;
    ledger_settle();
    const host_ledger_charge_request ledger_default{};
    const alloc_request              alloc_default{};
    expect(r, !ledger_default.may_yield, "host_ledger_charge_request defaults may_yield to false");
    expect(r, !alloc_default.may_yield, "alloc_request defaults may_yield to false");
    expect(r, alloc_default.load_txn_id == 0, "alloc_request defaults load_txn_id to 0");
    fake_yielder y;
    y.free_per_call = 10;
    if (!y.attach()) {
        void_arm(r, "the fake yielder could not register");
        return r;
    }
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(90).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    yieldable_chunks           chunks(2, 10);
    host_ledger_charge_request req{};
    req.bytes                            = 20;
    const host_ledger_charge_result c    = host_ledger_try_charge(req);
    const std::string               line = log_line_with("host-budget-exceeded");
    if (line.empty()) {
        void_arm(r, "no host-budget-exceeded line: the shortfall path did not run");
    }
    expect(r, !c.charged, "a request without may_yield is refused on a shortfall the yielders could cover");
    expect(r, y.calls.load() == 0, "the yielder is never called (calls=" + num(y.calls.load()) + ")");
    expect(r, printed(line, "yieldable_bytes") == 20, "the line logs yieldable_bytes=20");
    y.detach();
    ledger_settle();
    return r;
}

// (e) A yielder that frees nothing refuses after one empty round; one that frees too little refuses on the last round,
// one round per yieldable chunk counted at the first shortfall, with the retry count in the line.
// Engagement: the yielder was called, so the rounds ran.
// Mutant: an unbounded loop. It either spins (VOID at the 5 s bound, which fails) or keeps yielding until the ask fits.
arm_result arm_e() {
    arm_result r;
    ledger_settle();
    {
        fake_yielder y;
        y.free_per_call = 0;
        if (!y.attach()) {
            void_arm(r, "the fake yielder could not register");
            return r;
        }
        ggml_sycl_test_host_ledger_store_budget(100);
        if (!charge(90).charged) {
            void_arm(r, "the arm could not charge its baseline");
        }
        yieldable_chunks                chunks(2, 10);
        const host_ledger_charge_result c = bounded_charge(20, true, "a charge against a yielder that frees nothing");
        if (y.calls.load() == 0) {
            void_arm(r, "the yielder that frees nothing was never called");
        }
        const std::string line = log_line_with("host-budget-exceeded");
        expect(r, !c.charged, "a round that frees nothing refuses");
        expect(r, y.calls.load() == 1, "after one empty round (calls=" + num(y.calls.load()) + ")");
        expect(r, printed(line, "retries") == 1, "the line names retries=1");
        y.detach();
    }
    ledger_settle();
    {
        // 4 chunks counted at the first shortfall, so 4 rounds; each frees 10 of a deficit of 100.
        fake_yielder y;
        y.free_per_call = 10;
        if (!y.attach()) {
            void_arm(r, "the fake yielder could not register");
            return r;
        }
        ggml_sycl_test_host_ledger_store_budget(1000);
        if (!charge(1000).charged) {
            void_arm(r, "the arm could not charge its baseline");
        }
        yieldable_chunks                chunks(4, 100);
        const host_ledger_charge_result c =
            bounded_charge(100, true, "a charge against a yielder that frees too little");
        if (y.calls.load() == 0) {
            void_arm(r, "the yielder that frees too little was never called");
        }
        const std::string line = log_line_with("host-budget-exceeded");
        expect(r, !c.charged, "a yielder that frees too little refuses on the last round");
        expect(r, y.calls.load() == 4, "one round per counted chunk (calls=" + num(y.calls.load()) + ")");
        expect(r, c.retries == 4, "the result names 4 rounds (retries=" + num(c.retries) + ")");
        expect(r, printed(line, "retries") == 4, "the line names retries=4");
        y.detach();
    }
    ledger_settle();
    return r;
}

// (f) A yielder's own charge never yields: a charge made in the callback refuses at once, so the call count stays 1.
// Engagement: the callback ran and made its own charge.
// Mutant: no guard on a yield already running on the thread lets the inner charge yield (calls = 2).
arm_result arm_f() {
    arm_result r;
    ledger_settle();
    fake_yielder y;
    y.free_per_call = 10;
    y.inner_ask     = 25;  // a deficit of 15, under yieldable_bytes 20, so only the guard stops a second yield
    if (!y.attach()) {
        void_arm(r, "the fake yielder could not register");
        return r;
    }
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(90).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    yieldable_chunks                chunks(2, 10);
    const host_ledger_charge_result c = charge(20, true);
    if (!y.inner_tried.load()) {
        void_arm(r, "the callback never made its own charge");
    }
    expect(r, !y.inner_charged.load(), "the callback's own charge is refused");
    expect(r, y.calls.load() == 1, "the yielder is called once (calls=" + num(y.calls.load()) + ")");
    expect(r, c.charged, "the outer charge fits after its one round");
    y.detach();
    ledger_settle();
    return r;
}

// (g) The unregister handshake: the hook holds a yield between its in-flight increment and its callback load; the
// unregister waits for it; a sentinel set after the unregister returns is never seen by the yield.
// Engagement: the hook held the yield at YIELD_AFTER_INFLIGHT.
// Mutant: the callback loaded before the increment (the two statements swapped around the hook point): the unregister
// does not wait, and the yield calls the callback after the sentinel is set.
arm_result arm_g() {
    arm_result r;
    ledger_settle();
    fake_yielder y;
    y.free_per_call = 10;
    if (!y.attach()) {
        void_arm(r, "the fake yielder could not register");
        return r;
    }
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(90).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    yieldable_chunks chunks(2, 10);
    hold_arm(GGML_SYCL_TEST_HOST_LEDGER_POINT_YIELD_AFTER_INFLIGHT);
    if (!ggml_sycl_test_host_ledger_hook(GGML_SYCL_TEST_HOST_LEDGER_POINT_YIELD_AFTER_INFLIGHT, hold_hook)) {
        void_arm(r, "the YIELD_AFTER_INFLIGHT hook could not be set");
        y.detach();
        return r;
    }
    std::atomic<bool>         yield_done{ false };
    host_ledger_charge_result c{};
    std::thread               yt([&] {
        t_hold_this_thread = true;
        c                  = charge(20, true);
        yield_done.store(true);
    });
    if (!wait_for(g_hold.held)) {
        g_hold.release.store(true);
        yt.join();
        void_arm(r, "the hook never held the yield between its in-flight increment and its callback load");
        y.detach();
        ledger_settle();
        return r;
    }
    std::atomic<bool> unregister_done{ false };
    const int         slot = y.slot;
    std::thread       ut([&] {
        host_ledger_unregister_yielder(slot);
        y.destroyed.store(true);  // the sentinel: what a destroyed cache would be
        unregister_done.store(true);
    });
    y.slot = -1;  // the unregister is ut's
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool unregister_waited = !unregister_done.load();
    g_hold.release.store(true);
    yt.join();
    ut.join();
    if (g_hold.timed_out.load()) {
        void_arm(r, "the held yield was never released");
    }
    expect(r, unregister_waited, "the unregister waits for the yield that is in flight");
    expect(r, !y.saw_destroyed.load(), "the yield never sees the sentinel set after the unregister");
    expect(r, y.calls.load() == 0,
           "the yield finds the callback null and skips it (calls=" + num(y.calls.load()) + ")");
    expect(r, !c.charged, "with the only yielder gone the round frees nothing and the charge refuses");
    ledger_settle();
    return r;
}

// (h) A stored budget below committed refuses the next charge by name, a credit still lands, and committed neither
// wraps nor goes below 0; a committed + ask that would pass SIZE_MAX refuses.
// Engagement: the charges that set committed up succeeded, and each refusal logged its line.
// Mutants: an unsigned budget - committed admits the ask; the sum form committed + ask <= budget wraps and admits.
arm_result arm_h() {
    arm_result r;
    ledger_settle();
    ggml_sycl_test_host_ledger_store_budget(100);
    if (!charge(60).charged) {
        void_arm(r, "the arm could not charge its baseline");
    }
    ggml_sycl_test_host_ledger_store_budget(50);
    const host_ledger_charge_result c    = charge(1);
    const std::string               line = log_line_with("host-budget-exceeded");
    if (line.empty()) {
        void_arm(r, "no host-budget-exceeded line for the budget stored below committed");
    }
    expect(r, !c.charged, "a budget stored below committed refuses the next charge");
    expect(r, printed(line, "committed") == 60 && printed(line, "budget") == 50,
           "the refusal names committed=60 and budget=50");
    host_ledger_credit(10);
    expect(r, host_ledger_committed() == 50, "a credit still lands under a budget below committed");
    log_clear();
    host_ledger_credit(100);
    expect(r, host_ledger_committed() == 0, "a credit above committed saturates at 0, never wraps");
    expect(r, !log_line_with("[HOST-PLAN-BUG]").empty(), "a credit above committed logs [HOST-PLAN-BUG]");

    ledger_settle();
    ggml_sycl_test_host_ledger_store_budget(UINT64_MAX);
    const size_t near_max = SIZE_MAX - 10;
    if (!charge(near_max).charged) {
        void_arm(r, "the overflow case could not charge SIZE_MAX - 10");
    }
    const host_ledger_charge_result over = charge(20);
    expect(r, !over.charged, "a committed + ask past SIZE_MAX refuses");
    expect(r, host_ledger_committed() == near_max, "the refused overflow leaves committed alone");
    ledger_settle();
    return r;
}

// (i) The budget function saturates: a pinned share above total x pct, or an available reading at or below the OS
// reserve, gives a budget at or below committed that refuses by name, never a near-SIZE_MAX admit; an accepted
// GGML_SYCL_HOST_RESERVE_MB is exactly the budget; total and available both 0 give SIZE_MAX/4 with the WARN; total
// known and available 0 give committed, which refuses by name.
// Engagement: a positive control sample gives the formula's exact value, and each stored budget counts one store.
// Each refusing case is scored at its exact value, not at or below committed: a pinned share above the cap gives 0, and
// an available reading at or below the reserve gives committed, so the next charge refuses with nothing granted.
// Mutants: unsigned subtraction (either term) admits; the override replacing os_reserve only gives a sample-shaped
// budget instead of X; a failed sample treated as no budget at all (0, not committed) on a host whose total is known.
arm_result arm_i() {
    arm_result r;
    ledger_settle();
    const size_t total     = 64 * GiB;  // pct 100: pct_cap 64 GiB; os_reserve max(8 GiB, 2 GiB) = 8 GiB
    const size_t committed = 1 * GiB;

    host_budget_sample control{};
    control.total_bytes     = total;
    control.available_bytes = 40 * GiB;
    control.budget_pct      = 100;
    control.pinned_bytes    = 0;
    // min(64 - 0, 0 + (40 - 8)) GiB
    if (host_ledger_compute_budget(control, 0) != 32 * GiB) {
        void_arm(r, "the positive control does not give min(pct_cap - pinned, committed + available - os_reserve)");
    }
    host_budget_sample pct_half = control;
    pct_half.budget_pct         = 25;  // pct_cap 16 GiB
    expect(r, host_ledger_compute_budget(pct_half, 0) == 16 * GiB, "the pct term caps the budget at total x pct");

    uint64_t stores_before = 0;
    if (!ggml_sycl_test_host_ledger_stores(&stores_before)) {
        void_arm(r, "the store counter could not be read");
    }

    // Each refusing case: seed a budget, charge `committed`, store the computed budget, which must be `want`, and
    // charge one more byte.
    auto stored_refuses = [&](const host_budget_sample & s, size_t want, const char * what) {
        ledger_settle();
        ggml_sycl_test_host_ledger_store_budget(UINT64_MAX / 2);
        if (!charge(committed).charged) {
            void_arm(r, std::string(what) + ": could not charge committed");
        }
        const size_t stored = host_ledger_store_budget(s);
        expect(r, stored == want, std::string(what) + ": budget " + num(stored) + ", want " + num(want));
        expect(r, host_ledger_budget() == stored, std::string(what) + ": the computed budget is the stored one");
        const host_ledger_charge_result c = charge(1);
        expect(r, !c.charged, std::string(what) + ": the next charge refuses");
        expect(r, !log_line_with("host-budget-exceeded").empty(), std::string(what) + ": the refusal is named");
    };

    host_budget_sample pinned_over = control;
    pinned_over.pinned_bytes       = 80 * GiB;  // above pct_cap
    // min(64 - 80 saturating to 0, committed + 32 GiB)
    stored_refuses(pinned_over, 0, "a pinned share above total x pct");

    host_budget_sample avail_low = control;
    avail_low.available_bytes    = 6 * GiB;  // at or below os_reserve
    // min(64 GiB, committed + (6 - 8 saturating to 0))
    stored_refuses(avail_low, committed, "an available reading at or below os_reserve");

    host_budget_sample avail_zero = control;
    avail_zero.available_bytes    = 0;  // total known, the sample failed: not headroom
    // min(64 GiB, committed + (0 - 8 saturating to 0))
    stored_refuses(avail_zero, committed, "total known and available 0");

    uint64_t stores_after = 0;
    (void) ggml_sycl_test_host_ledger_stores(&stores_after);
    expect(r, stores_after - stores_before == 3, "each store_budget counts one store and the setter none");

    // The override is the whole budget: no pct, sample or share term enters.
    ledger_settle();
    host_budget_sample over = control;
    over.pinned_bytes       = 10 * GiB;
    over.reserve.status     = detail::host_reserve_parse_status::OVERRIDE;
    over.reserve.mb         = 3 * 1024;
    over.reserve.bytes      = 3 * GiB;
    expect(r, host_ledger_compute_budget(over, committed) == 3 * GiB,
           "an accepted GGML_SYCL_HOST_RESERVE_MB=3072 gives exactly 3 GiB");
    over.reserve.mb    = 100 * 1024;
    over.reserve.bytes = 100 * GiB;
    log_clear();
    expect(r, host_ledger_compute_budget(over, committed) == 100 * GiB,
           "an override above the computed budget still gives exactly the override");
    expect(r, log_count_with("GGML_SYCL_HOST_RESERVE_MB") == 1, "an override above the computed budget logs one WARN");

    // Both unknown: SIZE_MAX/4 and the WARN.
    log_clear();
    host_budget_sample unknown{};
    unknown.total_bytes     = 0;
    unknown.available_bytes = 0;
    expect(r, host_ledger_compute_budget(unknown, 0) == SIZE_MAX / 4, "total and available both 0 give SIZE_MAX/4");
    expect(r, !log_line_with("[HOST-ARENA] System RAM detection failed").empty(),
           "total and available both 0 log the detection WARN");

    // Total unknown: pct_cap is the sample and the reserve min(8 GiB, sample/4).
    host_budget_sample no_total{};
    no_total.total_bytes     = 0;
    no_total.available_bytes = 16 * GiB;  // reserve 4 GiB
    expect(r, host_ledger_compute_budget(no_total, 0) == 12 * GiB,
           "total unknown: min(available, committed + available - min(8 GiB, available/4))");
    ledger_settle();
    return r;
}

// --- the plan's host bytes (th32 §4.2): the two pure functions behind them, on hand-built plans ---

constexpr size_t MiB = 1024ull * 1024ull;

placement_plan plan_with_zones(size_t weight, size_t kv, size_t staging, size_t scratch, size_t max_tensor) {
    placement_plan plan;
    plan.entries.emplace_back();
    plan.host_zone_weight_bytes  = weight;
    plan.host_zone_kv_bytes      = kv;
    plan.host_zone_staging_bytes = staging;
    plan.host_zone_scratch_bytes = scratch;
    plan.max_tensor_bytes        = max_tensor;
    return plan;
}

// One plan's terms against the oracle's, term by term, so a failure names the term.
void expect_terms(arm_result & r, const char * name, const placement_plan & plan, size_t onednn_live, size_t chunk) {
    const host_zone_commit_terms                    t = placement_plan_host_zone_commit_terms(plan, onednn_live, chunk);
    const host_ledger_commit_oracle::derived_commit d =
        host_ledger_commit_oracle::derive_commit(plan, onednn_live, chunk);
    const std::string at = std::string(name) + ": ";
    expect(r, t.weight == d.weight, at + "weight " + num(t.weight) + ", the configure rules give " + num(d.weight));
    expect(r, t.kv == d.kv, at + "kv " + num(t.kv) + ", the configure rules give " + num(d.kv));
    expect(r, t.staging == d.staging,
           at + "staging " + num(t.staging) + ", the configure rules give " + num(d.staging));
    expect(r, t.scratch == d.scratch,
           at + "scratch " + num(t.scratch) + ", the configure rules give " + num(d.scratch));
    expect(r, t.runtime == d.runtime,
           at + "runtime " + num(t.runtime) + ", the configure rules give " + num(d.runtime));
    expect(r, t.zones() == t.weight + t.kv + t.staging + t.scratch, at + "zones() is not the four zones' sum");
    expect(r, t.total() == d.committed,
           at + "total() " + num(t.total()) + ", the pool grows by " + num(static_cast<size_t>(d.committed)));
}

// The commit terms are the bytes configuring the plan's zones commits on a fresh pool, term by term, and total() is
// the pool's growth. Three plans: one below both floors with every runtime term set, one whose 64-byte zone alignment
// carries it over a chunk boundary, and one with no entries.
// Engagement: the oracle's figures are nonzero, the first plan's runtime bytes take a chunk of their own, and the
// second plan's alignment adds a chunk its unaligned sum would not, so each mutant below moves a figure.
// Mutants: a term left out of total() or zones(); a floor dropped (the first plan's kv or scratch moves); no 64-byte
// alignment (the second plan loses a chunk); the runtime bytes folded into the zones' chunks (the first plan loses
// one); a plan with no entries charged.
arm_result leg_t() {
    arm_result     r;
    placement_plan floors              = plan_with_zones(5 * GiB + 7, 1 * MiB, 3 * MiB + 1, 0, 100 * MiB);
    floors.onednn_scratchpad_bytes     = 10 * MiB;
    floors.dma_staging_pool_bytes      = 5 * MiB;
    floors.pp_pipeline_scratch_bytes   = 2 * MiB;
    floors.pp_moe_onednn_scratch_bytes = 1 * MiB;
    const size_t floors_chunk          = 2 * GiB;
    const size_t floors_onednn         = 7 * MiB;

    const placement_plan aligned       = plan_with_zones(16 * MiB - 1, 64 * MiB, 63, 48 * MiB - 63, 0);
    const size_t         aligned_chunk = 16 * MiB;

    placement_plan empty = plan_with_zones(1 * GiB, 1 * GiB, 1 * GiB, 1 * GiB, 1 * GiB);
    empty.entries.clear();

    const host_ledger_commit_oracle::derived_commit df =
        host_ledger_commit_oracle::derive_commit(floors, floors_onednn, floors_chunk);
    const host_ledger_commit_oracle::derived_commit da =
        host_ledger_commit_oracle::derive_commit(aligned, 0, aligned_chunk);
    const size_t df_zone_chunks = (df.weight + df.kv + df.staging + df.scratch + floors_chunk - 1) / floors_chunk;
    const size_t da_unaligned   = (da.weight + da.kv + da.staging + da.scratch + aligned_chunk - 1) / aligned_chunk;
    if (df.committed == 0 || df.runtime == 0 || df.committed / (floors_chunk + 64) != df_zone_chunks + 1) {
        void_arm(r, "the first plan's runtime bytes do not take a chunk of their own");
        return r;
    }
    if (da.committed / (aligned_chunk + 64) != da_unaligned + 1) {
        void_arm(r, "the second plan's alignment adds no chunk, so the no-alignment mutant would survive");
        return r;
    }

    expect_terms(r, "below both floors", floors, floors_onednn, floors_chunk);
    expect_terms(r, "aligned across a chunk", aligned, 0, aligned_chunk);
    const host_zone_commit_terms none = placement_plan_host_zone_commit_terms(empty, 0, aligned_chunk);
    expect(r, none.total() == 0 && none.zones() == 0 && none.runtime == 0,
           "a plan with no entries is charged " + num(none.total()) + " bytes; configuring it commits nothing");
    return r;
}

// The zone commit check (th32 §4.2, §10's ledger check): configuring a plan's zones commits at most the bytes the plan
// counts. Growth equal to the plan, or below it on a pool that already held chunks, passes silently; growth past it
// returns false and logs one [HOST-PLAN-BUG] naming the device, the planned bytes and the committed bytes. Phase (a)
// checks the excess only; a pre-allocation that runs short is phase (b)'s check.
// Engagement: the three cases' inputs are the growths they name (equal, below and past the plan), so the excess case
// is the positive control the silent ones are read against.
// Mutants: the check removed (no line), the comparison inverted or made inclusive (a line on a silent case), a line
// that does not name both numbers or the device.
arm_result leg_c() {
    arm_result   r;
    const size_t planned = 100 * MiB;
    const size_t before  = 1 * GiB;
    const size_t equal   = before + planned;
    const size_t below   = before + 40 * MiB;
    const size_t past    = before + planned + 64;
    if (equal - before != planned || below - before >= planned || past - before <= planned) {
        void_arm(r, "the cases' growths are not equal to, below and past the plan");
        return r;
    }

    log_clear();
    const bool   equal_ok = host_zone_commit_check(3, planned, before, equal);
    const bool   below_ok = host_zone_commit_check(3, planned, before, below);
    const size_t silent   = log_count_with("[HOST-PLAN-BUG]");
    log_clear();
    const bool        past_ok = host_zone_commit_check(3, planned, before, past);
    const size_t      lines   = log_count_with("[HOST-PLAN-BUG]");
    const std::string line    = log_line_with("[HOST-PLAN-BUG]");

    expect(r, equal_ok && below_ok, "growth within the plan was refused");
    expect(r, silent == 0, num(silent) + " [HOST-PLAN-BUG] lines for growth within the plan");
    expect(r, !past_ok, "growth past the plan passed the check");
    expect(r, lines == 1, num(lines) + " [HOST-PLAN-BUG] lines for growth past the plan, want 1");
    expect(r, printed(line, "device") == 3, "the line does not name the device: " + line);
    expect(r, printed(line, "planned") == planned, "the line does not name the planned bytes: " + line);
    expect(r, printed(line, "committed") == past - before, "the line does not name the committed bytes: " + line);
    return r;
}

// --- a re-plan after the record (th32 §4.2, §10's zone check) ---

constexpr size_t KiB = 1024ull;

// The chunk the re-plan legs record with: small, so a growth of a few chunks crosses chunk boundaries and a stale
// record undercounts the pool by whole chunks.
constexpr size_t k_record_chunk = 64 * KiB;

// Records the plan's host figures as the planner's record does: the commit terms from the record's inputs (no live
// oneDNN scratch, the small chunk) and their total.
void record_terms(placement_plan & plan) {
    plan.host_zone_terms    = placement_plan_host_zone_commit_terms(plan, 0, k_record_chunk);
    plan.host_planned_bytes = plan.host_zone_terms.total();
}

// A plan the commit terms price: one entry, the four host zones above their floors, and no runtime bytes.
placement_plan replan_fixture_plan() {
    placement_plan plan{};
    plan.entries.emplace_back();
    plan.host_zone_weight_bytes  = 256 * MiB;
    plan.host_zone_kv_bytes      = 64 * MiB;
    plan.host_zone_staging_bytes = 16 * MiB;
    plan.host_zone_scratch_bytes = 64 * MiB;
    return plan;
}

// After a re-plan that grew a term: the re-plan logged no plan line and no would-refuse line, the plan holds the commit
// terms its own sizing gives from the record's inputs, term by term, and configuring it (on paper: the pool grows by
// those terms' total from an empty pool) logs no [HOST-PLAN-BUG]. The caller cleared the log before the re-plan.
void expect_rerecorded(arm_result & r, const std::string & name, const placement_plan & plan) {
    const std::string at = name + ": ";
    expect(r, log_count_with("would refuse") == 0,
           at + "the re-plan logged a would-refuse line: " + log_line_with("would refuse"));
    expect(r, log_count_with(" plan host_budget=") == 0,
           at + "the re-plan logged a plan line: " + log_line_with(" plan host_budget="));

    const host_zone_commit_terms   sized = placement_plan_host_zone_commit_terms(plan, 0, k_record_chunk);
    const host_zone_commit_terms & held  = plan.host_zone_terms;
    expect(r,
           held.weight == sized.weight && held.kv == sized.kv && held.staging == sized.staging &&
               held.scratch == sized.scratch && held.runtime == sized.runtime && held.rounding == sized.rounding,
           at + "the plan holds kv=" + num(held.kv) + " scratch=" + num(held.scratch) +
               " rounding=" + num(held.rounding) + ", its sizing gives kv=" + num(sized.kv) +
               " scratch=" + num(sized.scratch) + " rounding=" + num(sized.rounding));
    expect(r, plan.host_planned_bytes == sized.total(),
           at + "host_planned_bytes " + num(plan.host_planned_bytes) + ", its sizing's total " + num(sized.total()));

    log_clear();
    const bool configured_ok = host_zone_commit_check(0, held.total(), 0, sized.total());
    expect(r, configured_ok && log_count_with("[HOST-PLAN-BUG]") == 0,
           at + "configuring the re-planned plan names bytes it counts: " + log_line_with("[HOST-PLAN-BUG]"));
}

// The runtime transaction's KV update re-records the terms it grows (th32 §4.2). update_runtime_kv_sizes() and the KV
// refresh run on a recorded plan whose KV is on the host tier, as the runtime context transaction runs them on its copy
// of the published plan: the host KV grows past the KV zone the plan recorded, the refresh raises the zone, and the
// plan's terms and host_planned_bytes follow, so configuring it commits nothing the plan does not count.
// Engagement: the update raises the host KV zone past the recorded KV term by more than two chunks, so a stale record
// undercounts the pool by whole chunks.
// Mutants: the refresh not re-recording (the plan holds the old kv term and the check names the bytes); a re-record
// from the default or the cache's chunk instead of the record's (the rounding differs); a re-record through the
// planner's record (a plan line).
arm_result leg_k() {
    arm_result r;
    ggml_sycl_test_host_ledger_store_budget(64 * GiB);
    placement_plan plan = replan_fixture_plan();
    for (int layer = 0; layer < 4; ++layer) {
        plan.kv_device[layer] = -1;  // every layer's KV on the host tier
    }
    plan.kv_per_layer = 16 * MiB;
    plan.refresh_kv_byte_totals();  // 64 MiB of host KV, the KV zone the load plan sized
    record_terms(plan);
    const size_t recorded_kv = plan.host_zone_terms.kv;

    log_clear();
    plan.update_runtime_kv_sizes(/*n_ctx=*/8192, /*full_kv_per_layer=*/64 * MiB, /*swa_kv_per_layer=*/0);
    if (plan.kv_host_bytes != 256 * MiB || plan.host_zone_kv_bytes != plan.kv_host_bytes ||
        plan.host_zone_kv_bytes < recorded_kv + 2 * k_record_chunk) {
        void_arm(r, "the KV update did not raise the host KV zone past the recorded kv term " + num(recorded_kv) +
                        " by two chunks: host KV " + num(plan.kv_host_bytes) + ", zone " +
                        num(plan.host_zone_kv_bytes));
        ledger_settle();
        return r;
    }
    expect_rerecorded(r, "the KV refresh", plan);
    ledger_settle();
    return r;
}

// The runtime MMID re-plan re-records the terms its host pool grows (th32 §4.2). The re-plan runs on a recorded plan
// with one executing expert and no workspace yet, at the context's micro-batch: its demand's host pool goes into the
// SCRATCH zone (the growth tail of replan_moe_mmid_workspaces_for_runtime), and the plan's terms follow. Then the same
// zone grown by hand, outside any re-plan, is the positive control: configuring it names the bytes the plan does not
// count, so the silence after the re-plan is the re-record's and not the paper configure's.
// Engagement: the re-plan succeeds and grows SCRATCH by exactly its host pool, more than two chunks; the control's
// growth moves the same term.
// Mutants: the re-plan not re-recording, or re-recording before its write (the plan holds the old scratch term and the
// check names the bytes); a re-record from another chunk (the rounding differs); a re-record through the planner's
// record (a plan line). The configure step checking against its own recomputation instead of the recorded terms is the
// device fixture's mutant (plan-against-committed, device 1), since the configure step needs a cache.
arm_result leg_m() {
    arm_result r;
    ggml_sycl_test_host_ledger_store_budget(64 * GiB);
    static const char * k_expert = "blk.0.ffn_up_exps.weight";
    placement_plan      plan     = replan_fixture_plan();
    placement_entry &   expert   = plan.entries.back();
    expert.name                  = k_expert;
    expert.expert_id             = 0;
    expert.on_device             = true;
    expert.target_device         = 0;
    plan.device_id               = 0;
    plan.vram_budget             = 64 * GiB;
    plan.vram_bytes              = 1 * GiB;
    plan.planner_n_ubatch        = 4096;
    plan.planner_n_seq_max       = 1;
    record_terms(plan);
    const size_t scratch_before = plan.host_zone_scratch_bytes;

    placement_tensor_info info(k_expert, 4096ull * 1024ull * 8ull / 2ull);
    info.type                                          = GGML_TYPE_Q4_0;
    info.ne[0]                                         = 4096;
    info.ne[1]                                         = 1024;
    info.ne[2]                                         = 8;
    info.ne[3]                                         = 1;
    const std::vector<placement_tensor_info> inventory = { info };

    // The re-plan bounds the workspace owners by the device table's count: one mocked device, so SYCL stays untouched.
    ggml_sycl_device_info devices{};
    devices.device_count            = 1;
    devices.total_gpu_count         = 1;
    moe_mmid_runtime_reason reason  = moe_mmid_runtime_reason::OK;
    bool                    planned = false;
    log_clear();
    {
        test_sycl_info_override_guard                devices_guard(devices);
        test_moe_mmid_route_reachable_override_guard route_guard(/*reachable=*/true);
        planned = replan_moe_mmid_workspaces_for_runtime(plan, inventory, /*n_expert_used=*/4, &reason);
    }
    const size_t growth = plan.host_zone_scratch_bytes - scratch_before;
    if (!planned || plan.moe_mmid_host_pool_bytes == 0 || growth != plan.moe_mmid_host_pool_bytes ||
        growth < 2 * k_record_chunk) {
        void_arm(r, std::string("the re-plan did not grow SCRATCH by its host pool of at least two chunks: ok=") +
                        (planned ? "1" : "0") + " reason=" + moe_mmid_runtime_reason_name(reason) +
                        " host_pool=" + num(plan.moe_mmid_host_pool_bytes) + " growth=" + num(growth));
        ledger_settle();
        return r;
    }
    expect_rerecorded(r, "the MMID re-plan", plan);

    const size_t held_total = plan.host_zone_terms.total();
    plan.host_zone_scratch_bytes += 3 * k_record_chunk;  // outside any re-plan
    const size_t committed = placement_plan_host_zone_commit_terms(plan, 0, k_record_chunk).total();
    log_clear();
    const bool        control_ok = host_zone_commit_check(0, plan.host_zone_terms.total(), 0, committed);
    const size_t      lines      = log_count_with("[HOST-PLAN-BUG]");
    const std::string line       = log_line_with("[HOST-PLAN-BUG]");
    expect(r, !control_ok && lines == 1,
           "the control: a SCRATCH growth outside any re-plan gave " + num(lines) + " [HOST-PLAN-BUG] lines, want 1");
    expect(r, printed(line, "planned") == held_total && printed(line, "committed") == committed,
           "the control's line does not name the plan's " + num(held_total) + " and the pool's " + num(committed) +
               " bytes: " + line);
    ledger_settle();
    return r;
}

// --- the host sample's lock check (th32 §4.2) ---

// The text the witness abort carries, the probe the aborting case scores by.
constexpr const char * k_witness_message = "host_ledger_sample ran with a witnessed lock held";

struct child_run {
    bool        ran    = false;  // started, and exited or was signalled within the bound
    int         status = 0;
    std::string err;             // everything the child wrote to stderr
};

// Runs this binary as `--witness-child <mode>` and collects its stderr. The environment and argv are built before the
// fork, so the child calls only dup2, close and execve. The child runs with the witness checks at the build's default
// and with no abort backtrace.
child_run run_witness_child(const char * mode) {
    child_run                r;
    std::vector<std::string> env_strings;
    for (char ** e = environ; *e != nullptr; ++e) {
        if (std::strncmp(*e, "GGML_SYCL_WITNESS_CHECKS=", 25) != 0 && std::strncmp(*e, "GGML_NO_BACKTRACE=", 18) != 0) {
            env_strings.emplace_back(*e);
        }
    }
    env_strings.emplace_back("GGML_NO_BACKTRACE=1");
    std::vector<char *> envp;
    for (std::string & e : env_strings) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    char        arg0[] = "test-sycl-host-ledger";
    char        arg1[] = "--witness-child";
    std::string arg2   = mode;
    char *      argv[] = { arg0, arg1, arg2.data(), nullptr };

    int fds[2];
    if (pipe(fds) != 0) {
        return r;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        close(fds[1]);
        execve("/proc/self/exe", argv, envp.data());
        _exit(127);
    }
    close(fds[1]);
    if (pid < 0) {
        close(fds[0]);
        return r;
    }
    const auto deadline  = std::chrono::steady_clock::now() + k_bound;
    bool       timed_out = false;
    char       buf[512];
    for (;;) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            timed_out = true;
            kill(pid, SIGKILL);
            break;
        }
        pollfd    p     = { fds[0], POLLIN, 0 };
        const int ready = poll(&p, 1, static_cast<int>(left));
        if (ready <= 0) {
            continue;  // EINTR, or the bound passed: the loop's top decides
        }
        const ssize_t got = read(fds[0], buf, sizeof(buf));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            break;  // end of file: the child has exited
        }
        r.err.append(buf, static_cast<size_t>(got));
    }
    close(fds[0]);
    if (waitpid(pid, &r.status, 0) != pid) {
        return r;
    }
    r.ran = !timed_out;
    return r;
}

// The sample reads the host before any lock is taken (th32 §4.2): host_ledger_sample asserts that no witnessed lock is
// held other than the replan token, which a staged load holds and which ranks above every lock the pinned share takes.
// Each case runs in a child, since the assert aborts: one holding a witnessed L1 lock must abort with the witness
// message, and one holding only an L0 lock and one holding none must return.
// Engagement: each child reached the sample holding the witnessed locks its case names, with the witness checks on.
// Mutants: the assert removed (the L1 child returns); the assert counting the replan token too (the L0 child aborts).
arm_result leg_w() {
    arm_result r;

    struct witness_case {
        const char * mode;
        size_t       held;
        bool         aborts;
    };

    const witness_case cases[] = {
        { "l1",   1, true  },
        { "l0",   1, false },
        { "none", 0, false },
    };
    for (const witness_case & w : cases) {
        const child_run   c    = run_witness_child(w.mode);
        const std::string name = std::string("the ") + w.mode + " child";
        if (!c.ran) {
            void_arm(r, name + " could not be run or did not finish within the bound");
            continue;
        }
        const std::string reached = "witness-child reached held=" + num(w.held) + " enabled=1";
        if (c.err.find(reached) == std::string::npos) {
            void_arm(r, name + " did not reach the sample holding " + num(w.held) +
                            " witnessed lock(s) with the witness checks on: " + c.err);
            continue;
        }
        const bool aborted  = WIFSIGNALED(c.status) && WTERMSIG(c.status) == SIGABRT;
        const bool returned = WIFEXITED(c.status) && WEXITSTATUS(c.status) == 0 &&
                              c.err.find("witness-child returned") != std::string::npos;
        if (w.aborts) {
            expect(r, aborted,
                   name + " sampled with a witnessed lock held and did not abort (status " +
                       num(static_cast<size_t>(c.status)) + ")");
            expect(r, c.err.find(k_witness_message) != std::string::npos,
                   name + " did not print the witness message: " + c.err);
        } else {
            expect(
                r, returned,
                name + " did not return from the sample (status " + num(static_cast<size_t>(c.status)) + "): " + c.err);
        }
    }
    return r;
}

// The child of the sample leg. `mode` names the witnessed lock it holds while it samples: "l1", "l0" or "none".
int witness_child(const char * mode) {
    const bool         l1 = std::strcmp(mode, "l1") == 0;
    const bool         l0 = std::strcmp(mode, "l0") == 0;
    kv_witnessed_mutex lock(l1 ? KV_LOCK_L1_INVENTORY : KV_LOCK_L0_REPLAN, "the sample leg's lock");
    if (l1 || l0) {
        lock.lock();
    }
    std::fprintf(stderr, "witness-child reached held=%zu enabled=%d\n", kv_lock_witness::held_count(),
                 g_sycl_witness_enabled ? 1 : 0);
    std::fflush(stderr);
    (void) host_ledger_sample();
    std::fprintf(stderr, "witness-child returned\n");
    std::fflush(stderr);
    if (l1 || l0) {
        lock.unlock();
    }
    return 0;
}

struct arm_entry {
    const char * name;
    arm_result (*run)();
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--witness-child") == 0) {
        return witness_child(argv[2]);
    }

    ggml_log_callback prev_cb   = nullptr;
    void *            prev_data = nullptr;
    ggml_log_get(&prev_cb, &prev_data);
    ggml_log_set(capture_log, nullptr);

    const arm_entry arms[] = {
        { "a: the CAS race admits exactly one of two charges past the budget",                    arm_a },
        { "b: a deficit above yieldable_bytes refuses without a yield",                           arm_b },
        { "c: a shortfall one round covers fits with one yielder call",                           arm_c },
        { "d: may_yield defaults to false and the refusal logs yieldable_bytes",                  arm_d },
        { "e: the yield rounds are bounded by the chunks counted at the first shortfall",         arm_e },
        { "f: a yielder's own charge never yields",                                               arm_f },
        { "g: the unregister waits for the yield in flight",                                      arm_g },
        { "h: a budget below committed and an overflowing sum both refuse; the credit saturates", arm_h },
        { "i: the budget function saturates and the override is the whole budget",                arm_i },
        { "plan terms: the commit terms are what configuring the zones commits",                  leg_t },
        { "zone check: growth past the plan logs one [HOST-PLAN-BUG] naming both numbers",        leg_c },
        { "re-plan, KV: the KV refresh re-records the terms it grows",                            leg_k },
        { "re-plan, MMID: the runtime MMID re-plan re-records its host pool growth",              leg_m },
        { "sample: host_ledger_sample aborts with a witnessed lock held, the replan token aside", leg_w },
    };

    int n_pass = 0;
    int n_fail = 0;
    int n_void = 0;
    for (const arm_entry & arm : arms) {
        std::printf("(%s)\n", arm.name);
        const arm_result r = arm.run();
        switch (r.result) {
            case outcome::PASS:
                n_pass++;
                std::printf("  PASS\n");
                break;
            case outcome::FAIL:
                n_fail++;
                std::printf("  FAIL: %s\n", r.why.c_str());
                break;
            case outcome::VOID:
                n_void++;
                std::printf("  VOID: %s\n", r.why.c_str());
                break;
        }
        std::fflush(stdout);
    }

    clear_hooks();
    ggml_log_set(prev_cb, prev_data);
    std::printf("host-ledger arms: %d passed, %d failed, %d void\n", n_pass, n_fail, n_void);
    return n_fail == 0 && n_void == 0 ? 0 : 1;
}
