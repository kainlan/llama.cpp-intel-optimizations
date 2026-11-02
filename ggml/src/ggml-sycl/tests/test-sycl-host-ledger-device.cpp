//
// The process-global host ledger, arms (j)-(o) of th32 §11 T2, its phase (a) over-budget fixture and the real-pool leg
// of the plan-against-committed fixture: the budget's store order across planner passes and load transactions, the
// charge and credit at every exit of unified_allocate_owner, the must_host_cpu request rules, the yield's load
// exclusion, the tier-iff-token backstop, the one would-refuse WARN of a plan over the budget, and the zone commit
// check against the pool.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. Arm (j) runs the planner through real SYCL backends and stages a load, (m) and (o) make a device
// allocation from inside the HOST_CPU tier, and the pool-commit leg configures the host zones of a real cache. So the
// binary is registered gpu-serial, serial, with the selector pinned to the discrete cards, and a host with no SYCL GPU
// exits 77 before any arm runs.
//
// Every arm names its engagement check, the observation that proves it reached the mechanism it scores, and the
// mutant it kills. An arm whose engagement check fails is VOID, and a VOID arm fails the binary like a FAIL. Every arm
// runs, so one failure does not hide the rest. The order is fixed: the pool-commit leg needs caches whose host zones
// are not configured yet, and arm (j)'s late stage configures them, so (j) runs last.
//
// --strict-child A|B is the abort leg of arm (o), run by the parent in a child process with
// GGML_SYCL_STRICT_LEASES=1: it makes the backstop's request in mode A or B and must die by SIGABRT.

#include "../common.hpp"
#include "../kv-region-registry.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-commit-oracle.hpp"
#include "sycl-test-skip.hpp"

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_sycl;

namespace {

constexpr size_t MiB = 1024ull * 1024ull;

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

std::string num(uint64_t v) {
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

size_t log_count_with(const char * a, const char * b = nullptr) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    size_t                      n = 0;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos)) {
            ++n;
        }
    }
    return n;
}

// Every captured line that contains `a`, in order.
std::vector<std::string> log_lines_with(const char * a) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::vector<std::string>    out;
    for (const std::string & line : g_log_lines) {
        if (line.find(a) != std::string::npos) {
            out.push_back(line);
        }
    }
    return out;
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

// --- the hook board: one function serves every point, counting each call and firing an armed point ---

std::thread::id      g_test_thread;
std::atomic<int>     g_seen[GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT];
std::atomic<int>     g_armed[GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT];
std::atomic<int>     g_fired[GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT];
std::atomic<int>     g_arg_filter[GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT];
std::atomic<int32_t> g_force_mode{ 0 };
int32_t              nested_step();

// Only the test thread's calls count or fire, so a backend worker's allocation can neither consume an armed point nor
// move a count.
int32_t board_hook(int32_t point, int32_t arg) {
    if (point < 0 || point >= GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT || std::this_thread::get_id() != g_test_thread) {
        return 0;
    }
    g_seen[point].fetch_add(1, std::memory_order_acq_rel);
    if (point == GGML_SYCL_TEST_HOST_LEDGER_POINT_NESTED_HOST_CPU) {
        return nested_step();
    }
    const int filter = g_arg_filter[point].load(std::memory_order_acquire);
    if (filter >= 0 && filter != arg) {
        return 0;
    }
    int armed = g_armed[point].load(std::memory_order_acquire);
    while (armed > 0) {
        if (g_armed[point].compare_exchange_weak(armed, armed - 1, std::memory_order_acq_rel)) {
            g_fired[point].fetch_add(1, std::memory_order_acq_rel);
            return point == GGML_SYCL_TEST_HOST_LEDGER_POINT_FORCE_TIER_HOST_CPU ?
                       g_force_mode.load(std::memory_order_acquire) :
                       1;
        }
    }
    return 0;
}

void board_install() {
    for (int32_t p = 0; p < GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT; ++p) {
        g_seen[p].store(0);
        g_armed[p].store(0);
        g_fired[p].store(0);
        g_arg_filter[p].store(-1);
        (void) ggml_sycl_test_host_ledger_hook(p, board_hook);
    }
    g_force_mode.store(0);
}

void board_remove() {
    for (int32_t p = 0; p < GGML_SYCL_TEST_HOST_LEDGER_POINT_COUNT; ++p) {
        (void) ggml_sycl_test_host_ledger_hook(p, nullptr);
    }
}

void arm_point(int32_t point, int arg_filter = -1, int times = 1) {
    g_fired[point].store(0);
    g_arg_filter[point].store(arg_filter);
    g_armed[point].store(times);
}

void disarm_point(int32_t point) {
    g_armed[point].store(0);
    g_arg_filter[point].store(-1);
}

int seen(int32_t point) {
    return g_seen[point].load(std::memory_order_acquire);
}

int fired(int32_t point) {
    return g_fired[point].load(std::memory_order_acquire);
}

// --- the ledger as the exported accessors read it ---

struct reading {
    uint64_t committed = 0;
    uint64_t live      = 0;
    bool     ok        = false;
};

reading read_ledger() {
    reading r;
    r.ok = ggml_sycl_test_host_ledger_committed(&r.committed) && ggml_sycl_test_host_cpu_live_bytes(&r.live);
    return r;
}

uint64_t stores() {
    uint64_t n = 0;
    (void) ggml_sycl_test_host_ledger_stores(&n);
    return n;
}

// A budget no arm's charge reaches, so every owner-entry charge takes the CAS path and no would-refuse WARN is logged.
constexpr uint64_t k_roomy_budget = 1ull << 50;

alloc_request host_cpu_request(size_t bytes) {
    alloc_request req;
    req.device                           = 0;
    req.size                             = bytes;
    req.intent.role                      = alloc_role::LAZY_ROW_CACHE;
    req.intent.constraints.must_host_cpu = true;
    return req;
}

// A device request: no tier constraint, so the tier selection gives DEVICE_VRAM.
alloc_request device_request(size_t bytes) {
    alloc_request req;
    req.device = 0;
    req.size   = bytes;
    return req;
}

const char * const k_no_charge_line = "[HOST-CPU] unified_alloc refuses a HOST_CPU request with no charge";

// --- a yielder holding ledger chunks tagged with the load that made them ---

// Each chunk is a real charge on the ledger and one yieldable chunk. A call frees the first held chunk whose load is
// not the caller's, credits it and drops it from the yieldable sum: the exclusion a ring ensure makes for its own load
// (th32 §4.2).
struct tagged_yielder {
    struct chunk {
        uint64_t txn;
        size_t   bytes;
        bool     held;
    };

    std::vector<chunk>    chunks;
    std::atomic<int>      calls{ 0 };
    std::atomic<uint64_t> last_txn{ UINT64_MAX };
    int                   slot = -1;

    static size_t yield(void * ctx, size_t deficit, uint64_t load_txn_id) noexcept {
        (void) deficit;
        auto * self = static_cast<tagged_yielder *>(ctx);
        self->calls.fetch_add(1);
        self->last_txn.store(load_txn_id);
        for (chunk & c : self->chunks) {
            if (c.held && c.txn != load_txn_id) {
                c.held = false;
                host_ledger_credit(c.bytes);
                host_ledger_yieldable_sub(c.bytes);
                return c.bytes;
            }
        }
        return 0;
    }

    // Charges each chunk and registers the yielder; false when a chunk is empty, as nothing could be yielded from it,
    // or when a charge is refused. The chunks charged before that stay held, and the destructor credits them.
    bool hold(std::initializer_list<chunk> list) {
        for (const chunk & c : list) {
            if (c.bytes == 0) {
                return false;
            }
            host_ledger_charge_request req{};
            req.bytes = c.bytes;
            if (!host_ledger_try_charge(req).charged) {
                return false;
            }
            host_ledger_yieldable_add(c.bytes);
            chunks.push_back({ c.txn, c.bytes, true });
        }
        slot = host_ledger_register_yielder(&tagged_yielder::yield, this);
        return slot >= 0;
    }

    bool held(uint64_t txn) const {
        for (const chunk & c : chunks) {
            if (c.txn == txn) {
                return c.held;
            }
        }
        return false;
    }

    ~tagged_yielder() {
        host_ledger_unregister_yielder(slot);
        for (chunk & c : chunks) {
            if (c.held) {
                host_ledger_credit(c.bytes);
                host_ledger_yieldable_sub(c.bytes);
            }
        }
    }
};

// Every ledger arm starts with no hook armed, committed back where the arm found it and a budget it seeds itself.
struct ledger_baseline {
    uint64_t committed = 0;

    ledger_baseline() { committed = host_ledger_committed(); }

    // the arm's own charges, given back
    void restore() const {
        const uint64_t now = host_ledger_committed();
        if (now > committed) {
            host_ledger_credit(now - committed);
        }
    }
};

// The ledger's two figures, for a VOID that says why a hold did not fit.
std::string ledger_state() {
    return "committed " + num(host_ledger_committed()) + ", budget " + num(host_ledger_budget());
}

// (l) A class-C charge, made under a held witnessed lock with yieldable bytes the yielder could free, refuses and calls
// no yielder. Class C is every caller that may hold a lock, so its request carries may_yield = false, the default.
// The lock is a kv_witnessed_mutex at the inventory rank, the kind the host test's sample leg holds, so the witness
// sees it. The production lock a class-C charge runs under, g_tensor_inventory_mutex, is a plain mutex the witness
// does not see; holding a witnessed one means a witness check on the charge path, like the host sample's (design
// §4.2), would fire here instead of passing unseen.
// Engagement: while the charge runs the witness records the lock held with its checks on, and the same charge with
// may_yield = true and no lock held does call the yielder and fits, so the refusal is the class's, not an empty
// yielder's. Mutant: class C passing may_yield = true calls the yielder from under the lock.
arm_result arm_l_class_c_never_yields() {
    arm_result      r;
    ledger_baseline base;
    const uint64_t  B = base.committed + 100;
    ggml_sycl_test_host_ledger_store_budget(B);
    {
        tagged_yielder y;
        if (!y.hold({
                { 9, 60, true }
        })) {
            void_arm(r, "the yielder's chunk could not be charged");
        }
        expect(r, alloc_request{}.may_yield == false, "an alloc_request's may_yield does not default to false");
        kv_witnessed_mutex witnessed(KV_LOCK_L1_INVENTORY, "arm (l)'s class-C lock");
        const size_t       violations = kv_lock_witness::violations();
        {
            std::lock_guard<kv_witnessed_mutex> lock(witnessed);
            if (!g_sycl_witness_enabled || !kv_lock_witness::holds(KV_LOCK_L1_INVENTORY)) {
                void_arm(r, std::string("the witness does not see the held lock (checks ") +
                                (g_sycl_witness_enabled ? "on" : "off") + ")");
            }
            host_ledger_charge_request req{};
            req.bytes                             = 80;  // 40 over what is left, below the 60 the yielder holds
            const host_ledger_charge_result under = host_ledger_try_charge(req);
            expect(r, !under.charged, "a class-C charge over the budget was admitted");
            expect(r, y.calls.load() == 0, "a class-C charge called the yielder " + num(y.calls.load()) + " times");
        }
        expect(r, kv_lock_witness::violations() == violations,
               "the class-C charge recorded a witness violation: " + kv_lock_witness::last());
        host_ledger_charge_request req{};
        req.bytes                               = 80;
        req.may_yield                           = true;
        const host_ledger_charge_result control = host_ledger_try_charge(req);
        if (!control.charged || y.calls.load() != 1) {
            void_arm(r,
                     "the positive control (may_yield, no lock) did not yield and fit: calls " + num(y.calls.load()));
        }
    }
    base.restore();
    return r;
}

// (n) A charge that may not yield refuses with no yielder called, and a yield requested for a load skips that load's
// chunks while taking another load's. The row cache and the ring of th32 §3.3 are not in this tree, so the arm drives
// the shapes they would: a may_yield = false charge for the row cache's step 4, and a may_yield = true charge carrying
// a load_txn_id for the ring's ensure, first at the ledger and then through unified_allocate_owner, whose request
// fields must reach the charge. The row cache's own [HOST-PLAN-BUG] line belongs to the row cache and is not scored.
// Engagement: each leg seeds a budget its hold fits before it holds, the yielder holds a nonempty chunk of each load,
// and the deficit fits within them, so a yield is possible.
// Mutant: no load_txn_id exclusion (the ledger hands the yielder 0) frees the caller's own chunk.
arm_result arm_n_yield_excludes_own_load() {
    arm_result         r;
    ledger_baseline    base;
    constexpr uint64_t own = 7, other = 9;

    // the row cache's step 4: may not yield, refuses by name, calls nothing
    {
        // Each leg seeds its own budget before it holds: the budget the last arm or leg left counts that one's chunks.
        ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
        tagged_yielder y;
        if (!y.hold({
                { other, 60, true }
        })) {
            void_arm(r, "the step-4 leg's chunk could not be charged: " + ledger_state());
        }
        ggml_sycl_test_host_ledger_store_budget(host_ledger_committed() + 10);
        log_clear();
        host_ledger_charge_request req{};
        req.bytes                            = 50;
        const host_ledger_charge_result step = host_ledger_try_charge(req);
        expect(r, !step.charged, "the step-4 shortfall was admitted");
        expect(r, y.calls.load() == 0, "the step-4 shortfall called the yielder");
        expect(r, log_count_with("host-budget-exceeded", "deficit=40") == 1,
               "the step-4 refusal was not logged by name with its deficit");
    }

    // the ring's ensure, at the ledger
    {
        ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
        tagged_yielder y;
        if (!y.hold({
                { own,   40, true },
                { other, 40, true }
        })) {
            void_arm(r, "the ring leg's chunks could not be charged: " + ledger_state());
        }
        ggml_sycl_test_host_ledger_store_budget(host_ledger_committed());
        host_ledger_charge_request req{};
        req.bytes                            = 30;
        req.may_yield                        = true;
        req.load_txn_id                      = own;
        const host_ledger_charge_result ring = host_ledger_try_charge(req);
        if (y.calls.load() == 0) {
            void_arm(r, "the ring leg never reached the yielder");
        }
        expect(r, y.last_txn.load() == own,
               "the yielder was handed load " + num(y.last_txn.load()) + ", not " + num(own));
        expect(r, y.held(own), "the yield freed the caller's own chunk");
        expect(r, !y.held(other), "the yield did not take the other load's chunk");
        expect(r, ring.charged, "the ensure did not fit after the other load's chunk was freed");
        if (ring.charged) {
            host_ledger_credit(req.bytes);
        }
    }

    // the ring's ensure, through the owner entry: the request's may_yield and load_txn_id reach the charge
    {
        ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
        const size_t   ask = 64 * 1024;
        const size_t   m   = host_cpu_mapped_size(ask);
        tagged_yielder y;
        if (!y.hold({
                { own,   m, true },
                { other, m, true }
        })) {
            void_arm(r, "the owner leg's chunks of " + num(m) + " bytes could not be charged: " + ledger_state());
        }
        ggml_sycl_test_host_ledger_store_budget(host_ledger_committed());
        alloc_request req     = host_cpu_request(ask);
        req.may_yield         = true;
        req.load_txn_id       = own;
        allocation_result got = unified_allocate_owner(req);
        if (!got) {
            void_arm(r, "the owner leg's HOST_CPU request was refused");
        }
        if (y.calls.load() == 0) {
            void_arm(r, "the owner entry's charge never reached the yielder");
        }
        expect(r, y.last_txn.load() == own, "the owner entry handed the yielder load " + num(y.last_txn.load()));
        expect(r, y.held(own), "the owner entry's yield freed the caller's own chunk");
        expect(r, !y.held(other), "the owner entry's yield did not take the other load's chunk");
        got = {};
    }
    base.restore();
    return r;
}

// (k) After every exit of the owner entry, committed equals the charged bytes of the live HOST_CPU rows plus the
// bytes stranded with no row, and never wraps; each credited exit credits exactly once.
// Engagement: each exit's injection point fired, and the request was charged before it (the charge hook was seen),
// so the credit it is scored on is real. A base allocation stays live throughout, so a second credit would take
// committed below its prior value instead of saturating at 0 unseen.
// Mutants: the guard credits unconditionally (a record-freeing exit credits twice); the charge zeroed right after the
// emplace (the registry exit credits 0); the field missing from the release's detached copy (the retry credits 0);
// the credit before the free's result is known (the refused free credits, and the retry credits again).
arm_result arm_k_every_exit_accounts() {
    arm_result r;
    board_install();
    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
    const size_t ask = 64 * 1024;
    const size_t m   = host_cpu_mapped_size(ask);

    allocation_result base_alloc = unified_allocate_owner(host_cpu_request(ask));
    if (!base_alloc) {
        void_arm(r, "the base HOST_CPU allocation was refused");
        board_remove();
        return r;
    }
    uint64_t stranded  = 0;
    auto     invariant = [&](const std::string & where) {
        const reading x = read_ledger();
        if (!x.ok) {
            void_arm(r, where + ": the ledger accessors are unavailable");
            return x;
        }
        expect(r, x.committed < (UINT64_MAX >> 1), where + ": committed wrapped to " + num(x.committed));
        expect(r, x.committed == x.live + stranded,
                   where + ": committed " + num(x.committed) + " != live " + num(x.live) + " + stranded " + num(stranded));
        return x;
    };
    const reading prior = invariant("the base allocation");
    expect(r, prior.live >= m, "the base allocation's row carries no charge");

    struct exit_case {
        int32_t      point;
        int          arg;
        const char * name;
    };

    const exit_case credited[] = {
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_HOST_CPU_MMAP_FAIL,  -1, "mmap"                    },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_HOST_CPU_MLOCK_FAIL, -1, "mlock"                   },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_COORDINATOR_NULL,    -1, "coordinator null"        },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_CONTROL_NULL,        -1, "control null"            },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_OWNER_THROW,         0,  "throw before the record" },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_COHORT_FAIL,         -1, "cohort"                  },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_PUBLISH_FAIL,        -1, "publish"                 },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_REGISTRY_FAIL,       -1, "registry"                },
    };
    for (const exit_case & c : credited) {
        arm_point(c.point, c.arg);
        const int         charges = seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS);
        allocation_result got     = unified_allocate_owner(host_cpu_request(ask));
        disarm_point(c.point);
        if (fired(c.point) != 1) {
            void_arm(r, std::string("the ") + c.name + " exit never fired");
        }
        if (seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS) == charges) {
            void_arm(r, std::string("the ") + c.name + " request was not charged before its exit");
        }
        expect(r, !got, std::string("the ") + c.name + " exit returned an owner");
        const reading x = invariant(std::string("after the ") + c.name + " exit");
        expect(r, x.committed == prior.committed,
               std::string("the ") + c.name + " exit left committed " + num(x.committed) + ", prior " +
                   num(prior.committed) + ": not exactly one credit");
    }

    // The two exits that leave a control-less live row: the throw after unified_alloc and the identity mismatch.
    // Zero credit, the row in the live sum; then the row is removed, its bytes are stranded, and the test unmaps it and
    // gives the bytes back. The identity mismatch runs last, as the row asks.
    const exit_case rowed[] = {
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_OWNER_THROW,       1,  "throw after the record" },
        { GGML_SYCL_TEST_HOST_LEDGER_POINT_IDENTITY_MISMATCH, -1, "identity mismatch"      },
    };
    // FREE_REFUSED first: the release retry, and the stranded case at the three void exits.
    {
        allocation_result live = unified_allocate_owner(host_cpu_request(ask));
        if (!live) {
            void_arm(r, "the release-retry allocation was refused");
        } else {
            invariant("the release-retry allocation");
            arm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED);
            const release_attempt attempt = live.owner.reset();
            disarm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED);
            if (fired(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED) != 1) {
                void_arm(r, "the refused free never fired");
            }
            expect(r, !attempt.released(), "a refused free reported the allocation released");
            const reading held = invariant("after the refused free");
            expect(r, held.committed == prior.committed + m,
                   "the refused free changed committed: " + num(held.committed) + ", want " + num(prior.committed + m));
            auto         coordinator = unified_allocation_release_coordinator(0);
            const size_t released    = coordinator ? coordinator->process_retries() : 0;
            expect(r, released >= 1, "the retry released nothing");
            const reading after = invariant("after the release retry");
            expect(r, after.committed == prior.committed,
                   "the release retry left committed " + num(after.committed) + ", prior " + num(prior.committed));
        }
    }
    for (const exit_case & c : {
             exit_case{ GGML_SYCL_TEST_HOST_LEDGER_POINT_COHORT_FAIL,   -1, "cohort"   },
             exit_case{ GGML_SYCL_TEST_HOST_LEDGER_POINT_PUBLISH_FAIL,  -1, "publish"  },
             exit_case{ GGML_SYCL_TEST_HOST_LEDGER_POINT_REGISTRY_FAIL, -1, "registry" }
    }) {
        arm_point(c.point, c.arg);
        arm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED);
        allocation_result got = unified_allocate_owner(host_cpu_request(ask));
        disarm_point(c.point);
        disarm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED);
        if (fired(c.point) != 1 || fired(GGML_SYCL_TEST_HOST_LEDGER_POINT_FREE_REFUSED) != 1) {
            void_arm(r, std::string("the stranded ") + c.name + " exit did not fire both points");
        }
        expect(r, !got, std::string("the stranded ") + c.name + " exit returned an owner");
        stranded += m;
        const reading x = invariant(std::string("after the stranded ") + c.name + " exit");
        expect(r, x.live == prior.live, std::string("the stranded ") + c.name + " exit left a HOST_CPU row");
        // the mapping stays leaked (the injected refusal kept it, and no row names it); its bytes go back through the
        // ledger's own credit so the later exits start from the prior value
        host_ledger_credit(m);
        stranded -= m;
    }
    for (const exit_case & c : rowed) {
        const std::vector<void *> before = unified_alloc_registry_host_cpu_ptrs_for_test();
        arm_point(c.point, c.arg);
        allocation_result got = unified_allocate_owner(host_cpu_request(ask));
        disarm_point(c.point);
        if (fired(c.point) != 1) {
            void_arm(r, std::string("the ") + c.name + " exit never fired");
        }
        expect(r, !got, std::string("the ") + c.name + " exit returned an owner");
        const reading x = invariant(std::string("after the ") + c.name + " exit");
        expect(r, x.committed == prior.committed + m && x.live == prior.live + m,
               std::string("the ") + c.name + " exit credited its live row: committed " + num(x.committed));
        const std::vector<void *> after = unified_alloc_registry_host_cpu_ptrs_for_test();
        void *                    row   = nullptr;
        for (void * p : after) {
            if (std::find(before.begin(), before.end(), p) == before.end()) {
                row = p;
            }
        }
        if (row == nullptr) {
            void_arm(r, std::string("the ") + c.name + " exit left no row to remove");
            continue;
        }
        allocation_registry_test_erase(row);  // neither unmaps nor credits
        stranded += m;
        invariant(std::string("after removing the ") + c.name + " row");
        alloc_registry::instance().unregister_alloc(row);
        unified_cache_raw_free_host_cpu(row, m);
        host_ledger_credit(m);
        stranded -= m;
        invariant(std::string("after unmapping the ") + c.name + " row");
    }

    base_alloc        = {};
    const reading end = invariant("after the base allocation's release");
    expect(r, end.committed == prior.committed - m, "the base allocation's release did not credit it once");
    board_remove();
    return r;
}

// --- the nested topology of arm (m), driven from inside unified_alloc by the NESTED_HOST_CPU point ---

thread_local int  t_nested_depth = 0;
std::atomic<bool> g_nested_armed{ false };
std::atomic<int>  g_nested_at_depth1{ 0 };
std::atomic<int>  g_nested_at_depth2{ 0 };
std::atomic<int>  g_nested_owner_ok{ -1 };
std::atomic<int>  g_inner_result{ -1 };

// Depth 1 (inside the outer HOST_CPU owner entry's unified_alloc): a nested owner request that is not HOST_CPU, whose
// scope must null the charge. Depth 2 (inside that nested request's own unified_alloc): a direct must_host_cpu
// unified_alloc, which must refuse. The hook counts its own recursion; the point carries no depth.
int32_t nested_step() {
    if (!g_nested_armed.load()) {
        return 0;
    }
    if (t_nested_depth == 0) {
        ++t_nested_depth;
        g_nested_at_depth1.fetch_add(1);
        {
            allocation_result nested = unified_allocate_owner(device_request(4096));
            g_nested_owner_ok.store(nested ? 1 : 0);
        }
        --t_nested_depth;
        g_nested_armed.store(false);
        return 0;
    }
    if (t_nested_depth == 1) {
        ++t_nested_depth;
        g_nested_at_depth2.fetch_add(1);
        alloc_handle direct;
        const bool   ok = unified_alloc(host_cpu_request(4096), &direct);
        g_inner_result.store(ok ? 1 : 0);
        if (ok) {
            (void) unified_free(direct);
        }
        --t_nested_depth;
    }
    return 0;
}

// (m) unified_alloc refuses must_host_cpu with no charge held; a nested owner scope does not hand its caller's charge
// to a direct request; the seven routing fields beside must_host_cpu are rejected before the charge;
// forbid_vram_zone_spill is not one of them.
// Engagement: the direct refusal prints its named line; the nested topology fired at depth 1 and depth 2 and the
// nested device allocation succeeded; the positive control charges (the charge hook is seen) and credits once.
// Mutants: a refusal keyed on g_allocating_owner_control admits the depth-2 request (it then dies at the backstop
// with [HOST-PLAN-BUG] instead of the named line); a scope that does not null the thread-local lets the depth-2
// request take the outer charge and succeed; one dropped field check per row charges that row.
arm_result arm_m_request_rules() {
    arm_result r;
    board_install();
    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
    const size_t  ask   = 64 * 1024;
    const size_t  m     = host_cpu_mapped_size(ask);
    const reading prior = read_ledger();
    if (!prior.ok) {
        void_arm(r, "the ledger accessors are unavailable");
        board_remove();
        return r;
    }

    // the direct call with no charge
    log_clear();
    alloc_handle direct;
    expect(r, !unified_alloc(host_cpu_request(ask), &direct), "unified_alloc served must_host_cpu with no charge");
    if (log_count_with(k_no_charge_line) != 1) {
        void_arm(r, "the direct refusal did not print its named line");
    }
    expect(r, read_ledger().committed == prior.committed, "the direct refusal moved committed");

    // the nested topology
    log_clear();
    g_nested_at_depth1.store(0);
    g_nested_at_depth2.store(0);
    g_nested_owner_ok.store(-1);
    g_inner_result.store(-1);
    g_nested_armed.store(true);
    {
        allocation_result outer = unified_allocate_owner(host_cpu_request(ask));
        g_nested_armed.store(false);
        if (g_nested_at_depth1.load() != 1 || g_nested_at_depth2.load() != 1) {
            void_arm(r, "the nested topology did not fire at depth 1 and 2: " + num(g_nested_at_depth1.load()) + ", " +
                            num(g_nested_at_depth2.load()));
        }
        if (g_nested_owner_ok.load() != 1) {
            void_arm(r, "the nested device allocation was refused, so the depth-2 call never ran inside its scope");
        }
        expect(r, g_inner_result.load() == 0, "the depth-2 must_host_cpu request was served");
        expect(r, log_count_with(k_no_charge_line) == 1, "the depth-2 refusal did not print the named line");
        expect(r, log_count_with("[HOST-PLAN-BUG]") == 0, "the depth-2 request reached the tier backstop");
        expect(r, static_cast<bool>(outer), "the outer HOST_CPU allocation failed");
        const reading during = read_ledger();
        expect(r, during.committed == prior.committed + m,
               "the outer charge is not host_cpu_mapped_size of its ask: committed " + num(during.committed));
        expect(r, during.live == prior.live + m,
               "the outer row's charged_bytes is not host_cpu_mapped_size of its ask");
    }
    expect(r, read_ledger().committed == prior.committed, "the outer release did not credit exactly once");

    // the positive control: the plain request charges and credits once
    {
        const int         charges = seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS);
        allocation_result plain   = unified_allocate_owner(host_cpu_request(ask));
        if (!plain || seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS) == charges) {
            void_arm(r, "the positive control was refused or not charged");
        }
        expect(r, read_ledger().committed == prior.committed + m, "the positive control did not charge its mapping");
    }
    expect(r, read_ledger().committed == prior.committed, "the positive control's release did not credit once");

    struct rejected_row {
        const char * name;
        void (*set)(alloc_request &);
    };

    const rejected_row rows[] = {
        { "must_device",
         [](alloc_request & q) {
              q.intent.constraints.must_device = true;
          } },
        { "must_host_pinned",
         [](alloc_request & q) {
              q.intent.constraints.must_host_pinned = true;
          } },
        { "prefer_same_tier_as_cohort",
         [](alloc_request & q) {
              q.intent.constraints.prefer_same_tier_as_cohort = true;
          } },
        { "use_pinned_pool",
         [](alloc_request & q) {
              q.intent.constraints.use_pinned_pool = true;
          } },
        { "forbid_host_zone_growth",
         [](alloc_request & q) {
              q.intent.constraints.forbid_host_zone_growth = true;
          } },
        { "require_host_usm_base",
         [](alloc_request & q) {
              q.intent.constraints.require_host_usm_base = true;
          } },
        { "prefer_vram_zone",
         [](alloc_request & q) {
              q.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;
          } },
    };
    for (const rejected_row & row : rows) {
        alloc_request req = host_cpu_request(ask);
        row.set(req);
        const int         charges = seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS);
        allocation_result got     = unified_allocate_owner(req);
        expect(r, !got && got.error == allocation_error::INVALID_REQUEST,
               std::string("must_host_cpu with ") + row.name + " was not rejected as an invalid request");
        expect(r, seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS) == charges,
               std::string("must_host_cpu with ") + row.name + " was charged before its rejection");
        const reading x = read_ledger();
        expect(r, x.committed == prior.committed && x.live == prior.live,
               std::string("must_host_cpu with ") + row.name + " moved the ledger");
    }

    // forbid_vram_zone_spill is not a routing constraint: accepted, charged, credited once
    {
        alloc_request req                             = host_cpu_request(ask);
        req.intent.constraints.forbid_vram_zone_spill = true;
        allocation_result got                         = unified_allocate_owner(req);
        expect(r, static_cast<bool>(got), "must_host_cpu with forbid_vram_zone_spill was rejected");
        expect(r, read_ledger().committed == prior.committed + m, "forbid_vram_zone_spill's request was not charged");
    }
    expect(r, read_ledger().committed == prior.committed, "forbid_vram_zone_spill's release did not credit once");
    board_remove();
    return r;
}

// One backstop request in mode A (a device request forced to HOST_CPU, no charge held) or mode B (a HOST_CPU request
// forced to DEVICE_VRAM while its charge is held).
allocation_result backstop_request(char mode) {
    g_force_mode.store(mode == 'A' ? 1 : 2);
    arm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FORCE_TIER_HOST_CPU, mode == 'A' ? 0 : 1);
    allocation_result got = unified_allocate_owner(mode == 'A' ? device_request(4096) : host_cpu_request(64 * 1024));
    disarm_point(GGML_SYCL_TEST_HOST_LEDGER_POINT_FORCE_TIER_HOST_CPU);
    return got;
}

// (o) A cohort that holds a HOST_CPU row does not hand HOST_CPU to a prefer_same_tier_as_cohort request; the tier
// backstop refuses a tier that disagrees with the charge, in both directions, and aborts under strict.
// Engagement: the cohort's HOST_CPU row is live; the FORCE_TIER point fired with the expected charge argument and the
// [HOST-PLAN-BUG] line was printed; in mode B the request was charged (the charge hook was seen).
// Mutants: the cohort branch returns HOST_CPU (a variant build; it would refuse the follower at the backstop, so the
// follower's success and tier score it); the backstop dropped (mode A allocates a HOST_CPU mapping with no charge,
// mode B a device allocation with a charge held); the guard not crediting on the backstop's refusal (mode B leaves
// committed high).
arm_result arm_o_cohort_and_backstop() {
    arm_result r;
    board_install();
    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
    const reading prior = read_ledger();

    // the cohort leg
    {
        alloc_request first     = host_cpu_request(64 * 1024);
        first.intent.cohort_id  = "th32-t2-ledger-cohort";
        allocation_result owner = unified_allocate_owner(first);
        if (!owner) {
            void_arm(r, "the cohort's HOST_CPU row could not be made");
        }
        const reading with_row                                 = read_ledger();
        alloc_request follower                                 = device_request(4096);
        follower.intent.cohort_id                              = "th32-t2-ledger-cohort";
        follower.intent.constraints.prefer_same_tier_as_cohort = true;
        expect(r, unified_select_tier(follower) == alloc_tier::DEVICE_VRAM, "the cohort handed out HOST_CPU");
        allocation_result got = unified_allocate_owner(follower);
        expect(r, static_cast<bool>(got) && got.owner.metadata().tier != alloc_tier::HOST_CPU,
               "the follower was not served by normal selection");
        const reading after = read_ledger();
        expect(r, after.committed == with_row.committed && after.live == with_row.live,
               "the follower moved the ledger");
    }

    for (char mode : { 'A', 'B' }) {
        const std::string name    = std::string("mode ") + mode;
        const size_t      rows    = allocation_registry_test_size();
        const int         charges = seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS);
        log_clear();
        allocation_result got = backstop_request(mode);
        if (fired(GGML_SYCL_TEST_HOST_LEDGER_POINT_FORCE_TIER_HOST_CPU) != 1) {
            void_arm(r, name + ": the FORCE_TIER point did not fire with the expected charge argument");
        }
        if (mode == 'B' && seen(GGML_SYCL_TEST_HOST_LEDGER_POINT_CHARGE_BEFORE_CAS) == charges) {
            void_arm(r, name + ": the request was not charged before the backstop");
        }
        expect(r, !got && got.error == allocation_error::PHYSICAL_ALLOCATION_FAILED,
               name + ": the backstop did not give a typed refusal");
        expect(r, log_count_with("[HOST-PLAN-BUG]") == 1, name + ": the backstop printed no [HOST-PLAN-BUG] line");
        expect(r, allocation_registry_test_size() == rows, name + ": an allocation was made on some tier");
        const reading x = read_ledger();
        expect(r, x.live == prior.live, name + ": a HOST_CPU row was made");
        expect(r, x.committed == prior.committed,
               name + ": committed " + num(x.committed) + ", prior " + num(prior.committed) +
                   " (mode B credits its charge once)");
    }

    // the abort leg: the same request in a strict child process. The environment and argv are built before the fork,
    // so the child calls nothing but execve (this process has threads, and only async-signal-safe calls are legal
    // between fork and exec).
    std::vector<std::string> env_strings;
    for (char ** e = environ; *e != nullptr; ++e) {
        if (std::strncmp(*e, "GGML_SYCL_STRICT_LEASES=", 24) != 0 && std::strncmp(*e, "GGML_NO_BACKTRACE=", 18) != 0) {
            env_strings.emplace_back(*e);
        }
    }
    env_strings.emplace_back("GGML_SYCL_STRICT_LEASES=1");
    env_strings.emplace_back("GGML_NO_BACKTRACE=1");
    std::vector<char *> envp;
    for (std::string & e : env_strings) {
        envp.push_back(e.data());
    }
    envp.push_back(nullptr);
    for (const char * mode : { "A", "B" }) {
        char        arg0[] = "test-sycl-host-ledger-device";
        char        arg1[] = "--strict-child";
        std::string arg2   = mode;
        char *      argv[] = { arg0, arg1, arg2.data(), nullptr };
        const pid_t pid    = fork();
        if (pid == 0) {
            execve("/proc/self/exe", argv, envp.data());
            _exit(127);
        }
        int status = 0;
        if (pid < 0 || waitpid(pid, &status, 0) != pid) {
            void_arm(r, std::string("the strict child for mode ") + mode + " could not be run");
            continue;
        }
        expect(r, WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
               std::string("the strict child for mode ") + mode + " did not abort (status " + num(status) + ")");
    }
    board_remove();
    return r;
}

// The child of arm (o)'s abort leg. Returning at all is the failure.
int strict_child(const char * mode) {
    g_test_thread = std::this_thread::get_id();
    ggml_log_set(capture_log, nullptr);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_sycl_init(0), ggml_backend_free);
    if (!backend) {
        std::fprintf(stderr, "FAIL: the strict child could not initialise backend 0\n");
        return 4;
    }
    board_install();
    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
    allocation_result got = backstop_request(mode[0]);
    std::fprintf(stderr, "FAIL: the strict backstop returned (owner %d)\n", got ? 1 : 0);
    return 3;
}

// --- a Q2_K-shaped MoE inventory, small enough for a test, in the planner's form and the loader's ---

struct inventory_row {
    std::string name;
    ggml_type   type;
    int64_t     ne[3];
};

constexpr int     k_layers  = 2;
constexpr int     k_experts = 8;
constexpr int64_t k_embd    = 1024;
constexpr int64_t k_ffn     = 512;
constexpr int64_t k_vocab   = 4096;

std::vector<inventory_row> q2k_rows() {
    std::vector<inventory_row> rows;
    rows.push_back({
        "token_embd.weight", GGML_TYPE_Q2_K, { k_embd, k_vocab, 1 }
    });
    for (int l = 0; l < k_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        rows.push_back({
            blk + "attn_norm.weight", GGML_TYPE_F32, { k_embd, 1, 1 }
        });
        rows.push_back({
            blk + "attn_q.weight", GGML_TYPE_Q2_K, { k_embd, k_embd, 1 }
        });
        rows.push_back({
            blk + "attn_k.weight", GGML_TYPE_Q2_K, { k_embd, k_embd, 1 }
        });
        rows.push_back({
            blk + "attn_v.weight", GGML_TYPE_Q2_K, { k_embd, k_embd, 1 }
        });
        rows.push_back({
            blk + "attn_output.weight", GGML_TYPE_Q2_K, { k_embd, k_embd, 1 }
        });
        rows.push_back({
            blk + "ffn_norm.weight", GGML_TYPE_F32, { k_embd, 1, 1 }
        });
        rows.push_back({
            blk + "ffn_gate_inp.weight", GGML_TYPE_F32, { k_embd, k_experts, 1 }
        });
        rows.push_back({
            blk + "ffn_gate_exps.weight", GGML_TYPE_Q2_K, { k_embd, k_ffn, k_experts }
        });
        rows.push_back({
            blk + "ffn_up_exps.weight", GGML_TYPE_Q2_K, { k_embd, k_ffn, k_experts }
        });
        rows.push_back({
            blk + "ffn_down_exps.weight", GGML_TYPE_Q2_K, { k_ffn, k_embd, k_experts }
        });
    }
    rows.push_back({
        "output_norm.weight", GGML_TYPE_F32, { k_embd, 1, 1 }
    });
    rows.push_back({
        "output.weight", GGML_TYPE_Q2_K, { k_embd, k_vocab, 1 }
    });
    return rows;
}

size_t row_bytes(const inventory_row & row) {
    return ggml_row_size(row.type, row.ne[0]) * static_cast<size_t>(row.ne[1]) * static_cast<size_t>(row.ne[2]);
}

std::vector<placement_tensor_info> planner_inventory() {
    std::vector<placement_tensor_info> inv;
    for (const inventory_row & row : q2k_rows()) {
        placement_tensor_info t(row.name, row_bytes(row));
        t.type  = row.type;
        t.ne[0] = row.ne[0];
        t.ne[1] = row.ne[1];
        t.ne[2] = row.ne[2];
        t.ne[3] = 1;
        inv.push_back(t);
    }
    return inv;
}

// The loader's form. The rows own the name strings the tensor infos point at.
struct loader_inventory {
    std::vector<inventory_row>         rows = q2k_rows();
    std::vector<ggml_sycl_tensor_info> tensors;
    ggml_sycl_tensor_inventory         inv{};

    loader_inventory() {
        size_t total = 0;
        for (const inventory_row & row : rows) {
            ggml_sycl_tensor_info t{};
            t.name  = row.name.c_str();
            t.size  = row_bytes(row);
            t.type  = row.type;
            t.ne[0] = row.ne[0];
            t.ne[1] = row.ne[1];
            t.ne[2] = row.ne[2];
            t.ne[3] = 1;
            total += t.size;
            tensors.push_back(t);
        }
        inv.tensors       = tensors.data();
        inv.count         = tensors.size();
        inv.total_size    = total;
        inv.n_expert      = k_experts;
        inv.n_expert_used = 2;
        inv.n_layer       = k_layers;
        inv.n_ctx         = 16;
        inv.n_ubatch      = 1;
    }
};

// The over-budget fixture of th32 §11 T2, phase (a): a plan whose host bytes exceed the stored budget logs one
// would-refuse WARN carrying the plan's number and each of its terms, and the load proceeds: the plan comes back whole
// and nothing is refused.
// Engagement: the starved plan put weights on the host, its [HOST-LEDGER] plan line was logged, and the same plan under
// a roomy budget logs no WARN, so the WARN is the budget's doing.
// Mutants: the WARN dropped (no line); one WARN per term (several lines); a term or the plan's number missing from the
// line; the plan's host_budget read before the store (the plan records another budget); a refusal in phase (a).
arm_result fixture_over_budget_warns_and_loads() {
    arm_result r;
    if (get_unified_cache_for_device(0) == nullptr) {
        void_arm(r, "device 0 has no cache, so the plan's chunk figure is not the pool's");
        return r;
    }
    const std::vector<placement_tensor_info> inventory = planner_inventory();
    const uint64_t                           tiny      = 4096;
    placement_kv_info                        kv{};

    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);
    log_clear();
    const placement_plan           roomy       = compute_placement_plan(inventory, 1 * MiB, 0, kv, nullptr, k_experts);
    const std::vector<std::string> roomy_warns = log_lines_with("would refuse: host-budget-exceeded");

    ggml_sycl_test_host_ledger_store_budget(tiny);
    log_clear();
    const placement_plan           plan    = compute_placement_plan(inventory, 1 * MiB, 0, kv, nullptr, k_experts);
    const std::vector<std::string> warns   = log_lines_with("would refuse: host-budget-exceeded");
    const std::vector<std::string> refused = log_lines_with("host-budget-exceeded");
    const std::vector<std::string> lines   = log_lines_with(" plan host_budget=");
    ggml_sycl_test_host_ledger_store_budget(k_roomy_budget);

    if (plan.entries.empty() || plan.weight_host_bytes == 0 || roomy.weight_host_bytes == 0) {
        void_arm(r, "the starved plan placed nothing on the host");
        return r;
    }
    if (lines.size() != 1) {
        void_arm(r, num(lines.size()) + " [HOST-LEDGER] plan lines for one plan, want 1");
        return r;
    }
    expect(r, roomy_warns.empty(), "a roomy budget logged " + num(roomy_warns.size()) + " would-refuse lines");
    expect(r, plan.host_budget == tiny,
           "the plan recorded host_budget " + num(plan.host_budget) + ", want " + num(tiny));
    expect(
        r, plan.host_planned_bytes > tiny,
        "the plan's host bytes " + num(plan.host_planned_bytes) + " do not exceed the budget, so nothing is over it");
    expect(r, warns.size() == 1, num(warns.size()) + " would-refuse lines, want 1");
    expect(r, refused.size() == warns.size(), "phase (a) logged a refusal beside the would-refuse WARN");
    if (warns.size() == 1) {
        const std::string &            w = warns[0];
        const host_zone_commit_terms & t = plan.host_zone_terms;
        expect(r, printed(w, "host_budget") == tiny, "the WARN does not name the budget: " + w);
        expect(r, printed(w, "host_planned_bytes") == plan.host_planned_bytes,
               "the WARN does not name the plan's number: " + w);
        expect(r, printed(w, "weight") == t.weight, "the WARN does not name the weight term: " + w);
        expect(r, printed(w, "kv") == t.kv, "the WARN does not name the kv term: " + w);
        expect(r, printed(w, "staging") == t.staging, "the WARN does not name the staging term: " + w);
        expect(r, printed(w, "scratch") == t.scratch, "the WARN does not name the scratch term: " + w);
        expect(r, printed(w, "runtime") == t.runtime, "the WARN does not name the runtime term: " + w);
        expect(r, printed(w, "rounding") == t.rounding, "the WARN does not name the rounding term: " + w);
    }
    return r;
}

// The pool-commit leg of the plan-against-committed fixture (th32 §4.2, §10's ledger check). On device 0 the plan's
// zone terms are the configure rules' bytes, its host_planned_bytes is their total, and configuring its zones commits
// exactly that, with no [HOST-PLAN-BUG]; on device 1 the plan's runtime grows by one chunk after planning, so
// configuring it commits one pool chunk the plan does not count, and the check names both numbers. On both devices the
// real plan's CPU quant buffer bytes are in no term (design §4.2, "Two terms are deliberately absent"): the terms equal
// those of the same plan with the field zeroed. The PP MoE oneDNN scratch ring is 0 on a plan built by a direct planner
// call, because its planned bytes are published by the load path, so this leg does not engage it; the host leg's
// hand-built plan does.
// Engagement: both caches are fresh (no chunk, no zones), the derived figure is nonzero, the real plan's CPU quant, DMA
// staging and oneDNN scratchpad fields and its RUNTIME pool term are nonzero, and the delta is measured on the real
// pool. Mutants: a term dropped from the plan's number (the equality fails on device 0, and the bug line's planned
// figure moves on device 1); cpu_quant_buffer_bytes added to a term (the runtime or a zone term differs from the
// configure rules' and from the zeroed copy's); the check removed (no line on device 1); the check comparing
// configure's own recomputation instead of the plan's recorded terms (no line on device 1).
arm_result fixture_plan_against_committed() {
    arm_result r;
    // Two cards, counted before the scheduler hides any: the scheduler exposes device 0 alone unless a split is asked
    // for, and the leg plans and configures each card's own cache, as the multi-device planner does for a hidden card.
    const int  n_gpu = ggml_sycl_info().total_gpu_count;
    if (n_gpu < 2) {
        void_arm(r, "the leg needs two SYCL GPUs (selector level_zero:0,1), found " + num(n_gpu));
        return r;
    }
    const std::vector<placement_tensor_info> inventory = planner_inventory();
    for (int device = 0; device < 2; ++device) {
        const std::string name  = "device " + std::to_string(device);
        unified_cache *   cache = get_unified_cache_for_device(device);
        if (cache == nullptr || cache->pinned_pool_budget() == 0) {
            void_arm(r, name + ": no cache with a pinned pool");
            continue;
        }
        if (cache->pinned_pool_chunk_count() != 0 || cache->host_zones_configured()) {
            void_arm(r, name + ": the cache's pool is not fresh, so its delta is not the plan's figure");
            continue;
        }
        placement_kv_info kv{};
        placement_plan    plan =
            compute_placement_plan(inventory, /*vram_budget=*/1 * MiB, device, kv, nullptr, k_experts);
        if (plan.entries.empty() || plan.weight_host_bytes == 0) {
            void_arm(r, name + ": the starved plan placed nothing on the host");
            continue;
        }
        const size_t chunk       = cache->pinned_pool_chunk_size();
        const size_t onednn_live = cache->onednn_weights_scratch_size() + cache->onednn_activations_scratch_size();
        const host_ledger_commit_oracle::derived_commit d =
            host_ledger_commit_oracle::derive_commit(plan, onednn_live, chunk);
        const host_zone_commit_terms & t       = plan.host_zone_terms;
        const uint64_t                 planned = t.total();
        if (d.committed == 0) {
            void_arm(r, name + ": the configure rules derive no bytes for the plan");
            continue;
        }
        if (plan.cpu_quant_buffer_bytes == 0 || plan.dma_staging_pool_bytes == 0 || plan.onednn_scratchpad_bytes == 0 ||
            t.runtime == 0) {
            void_arm(r, name + ": the real plan's CPU quant " + num(plan.cpu_quant_buffer_bytes) + ", DMA staging " +
                            num(plan.dma_staging_pool_bytes) + ", oneDNN scratchpad " +
                            num(plan.onednn_scratchpad_bytes) + " or runtime term " + num(t.runtime) +
                            " is 0, so the CPU quant exclusion is not engaged");
            continue;
        }
        expect(r,
               t.weight == d.weight && t.kv == d.kv && t.staging == d.staging && t.scratch == d.scratch &&
                   t.runtime == d.runtime,
               name + ": the plan's zone terms are not the configure rules' bytes");
        placement_plan no_quant         = plan;
        no_quant.cpu_quant_buffer_bytes = 0;
        const host_zone_commit_terms t0 = placement_plan_host_zone_commit_terms(no_quant, onednn_live, chunk);
        expect(r,
               t.weight == t0.weight && t.kv == t0.kv && t.staging == t0.staging && t.scratch == t0.scratch &&
                   t.runtime == t0.runtime && t.total() == t0.total(),
               name + ": the CPU quant buffer bytes " + num(plan.cpu_quant_buffer_bytes) + " moved a term: runtime " +
                   num(t.runtime) + " against " + num(t0.runtime));
        expect(r, planned == d.committed,
               name + ": the plan's zone terms total " + num(planned) + ", the configure rules commit " +
                   num(d.committed));
        expect(r, plan.host_planned_bytes == planned,
               name + ": host_planned_bytes " + num(plan.host_planned_bytes) + " is not the zone terms' total " +
                   num(planned));
        if (device == 1) {
            plan.pp_pipeline_scratch_bytes += chunk;  // one more runtime chunk, outside the plan's figure
        }
        const uint64_t before = cache->pinned_pool_committed();
        log_clear();
        configure_host_zones_for_plan_for_testing(cache, plan);
        const uint64_t delta = cache->pinned_pool_committed() - before;
        if (!cache->host_zones_configured()) {
            void_arm(r, name + ": the zones were not configured, so nothing was measured");
            continue;
        }
        const std::vector<std::string> bugs = log_lines_with("[HOST-PLAN-BUG]");
        if (device == 0) {
            expect(r, delta == planned,
                   name + ": configuring committed " + num(delta) + ", the plan says " + num(planned));
            expect(r, bugs.empty(), name + ": a [HOST-PLAN-BUG] line on a plan that counts every chunk");
        } else {
            expect(r, delta == planned + chunk + 64,
                   name + ": the uncounted chunk was not committed: delta " + num(delta));
            expect(r, bugs.size() == 1, name + ": " + num(bugs.size()) + " [HOST-PLAN-BUG] lines, want 1");
            if (bugs.size() == 1) {
                expect(r, printed(bugs[0], "planned") == planned,
                       name + ": the bug line does not name the planned bytes");
                expect(r, printed(bugs[0], "committed") == delta,
                       name + ": the bug line does not name the committed bytes");
            }
        }
    }
    return r;
}

// The host_budget each plan recorded, in the order the plans were built.
std::vector<size_t> plan_budgets() {
    std::vector<size_t> out;
    for (const std::string & line : log_lines_with("[HOST-LEDGER] device=")) {
        if (line.find(" plan host_budget=") != std::string::npos) {
            out.push_back(printed(line, "host_budget"));
        }
    }
    return out;
}

bool all_equal(const std::vector<size_t> & v, size_t want) {
    for (size_t x : v) {
        if (x != want) {
            return false;
        }
    }
    return !v.empty();
}

std::string list(const std::vector<size_t> & v) {
    std::string s;
    for (size_t x : v) {
        s += (s.empty() ? "" : ",") + num(x);
    }
    return s.empty() ? "(none)" : s;
}

// (j) The store order. A planner pass stores before it plans, so its plan records the new budget; one load
// transaction stores once across its early and late stage runs and every plan they build; a second transaction stores
// again; a direct entry outside a transaction stores, and one while a transaction is open only reads.
// A stage run builds one plan per device the scheduler exposes. On a multi-GPU host that is device 0 alone unless a
// split is asked for, and its plan is the multi-device plan over every card, so the arm runs on any number of cards.
// Engagement: every leg's plans were built (their [HOST-LEDGER] plan lines are counted, one per exposed device in a
// stage run), the stage runs returned OK, and the pinned commit injected between the early and the late run moves the
// budget by exactly the bytes the pool committed, read back: the budget computed from one sample with the pinned bytes
// before and after the commit differs by those bytes, and the one before is the stored value, so a second store would
// have recorded the stored value less the commit.
// Mutants: the store after the plan call (the first leg's plan records B0); a store per stage call (stores reads 2
// and the late plans read the early value less the injected bytes); a store per planner call, inside the stage run's
// device loop (the transaction stores at least once per stage run, so twice, and the late plans read the early value
// less the injected bytes); no current_active_token() check (the direct entry inside the transaction stores); id 0
// writing the stored transaction (the second direct entry skips its store). A dry run is not in this tree, so the
// last leg is the planner called outside any scope, which must read B0 and store nothing.
arm_result arm_j_store_order() {
    arm_result   r;
    const size_t n_dev = static_cast<size_t>(ggml_backend_sycl_get_device_count());  // the stage runs' device loop
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_sycl_init(0), ggml_backend_free);
    if (!backend) {
        void_arm(r, "backend 0 did not initialise");
        return r;
    }
    loader_inventory inv;
    const uint64_t   B0 = 4096;

    // a planner pass with a prior stored B0 records the new value
    ggml_sycl_test_host_ledger_store_budget(B0);
    uint64_t s0 = stores();
    log_clear();
    ggml_backend_sycl_compute_placement_plan_early(backend.get(), &inv.inv);
    std::vector<size_t> plans = plan_budgets();
    if (plans.empty()) {
        void_arm(r, "the first direct entry built no plan");
    }
    const size_t N1 = host_ledger_budget();
    expect(r, stores() == s0 + 1, "a direct entry outside a transaction stored " + num(stores() - s0) + " times");
    expect(r, N1 != B0, "the direct entry's store left B0");
    expect(r, all_equal(plans, N1), "the first plan recorded " + list(plans) + ", not the new budget " + num(N1));

    // two consecutive direct entries store twice
    s0 = stores();
    ggml_backend_sycl_compute_placement_plan_early(backend.get(), &inv.inv);
    ggml_backend_sycl_compute_placement_plan_early(backend.get(), &inv.inv);
    expect(r, stores() == s0 + 2, "two direct entries stored " + num(stores() - s0) + " times, want 2");

    // one transaction: a direct entry reads, the early and late stage runs store once between them
    ggml_sycl_load_txn txn{};
    if (ggml_backend_sycl_model_load_begin(&txn) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the load transaction did not begin");
        return r;
    }
    s0 = stores();
    log_clear();
    ggml_backend_sycl_compute_placement_plan_early(backend.get(), &inv.inv);
    plans = plan_budgets();
    if (plans.empty()) {
        void_arm(r, "the direct entry inside the transaction built no plan");
    }
    expect(r, stores() == s0, "a direct entry while a transaction is open stored");
    expect(r, all_equal(plans, host_ledger_budget()), "the direct entry inside the transaction did not read the store");

    log_clear();
    const ggml_sycl_lifecycle_result early_rc = ggml_backend_sycl_stage_inventory_plan(&inv.inv, nullptr, true);
    const std::vector<size_t>        early    = plan_budgets();
    const size_t                     S1       = host_ledger_budget();
    if (early_rc != GGML_SYCL_LIFECYCLE_OK || early.size() < n_dev) {
        void_arm(r, "the early stage run returned " + num(early_rc) + " with " + num(early.size()) +
                        " plans, want one per exposed device (" + num(n_dev) + ")");
    }
    expect(r, stores() == s0 + 1, "the early stage run stored " + num(stores() - s0) + " times, want 1");
    expect(r, all_equal(early, S1), "the early plans recorded " + list(early) + ", stored " + num(S1));

    // A pinned commit between the runs. The injection grows device 0's RUNTIME chunks by the shortfall past the free
    // bytes they already hold, each chunk pinned host memory committed to the pool and never charged to the ledger, so
    // the budget sees it through the pinned share. In this fixture the share is the binding term: 5 % of RAM less the
    // pinned bytes sits far below committed plus MemAvailable less the OS reserve. So the commit moves the budget by
    // the bytes the pool committed, which are read back here, not by the bytes requested. The two budgets come from
    // one sample that differs only in its pinned bytes, so a MemAvailable reading cannot move between them.
    constexpr size_t requested     = 512 * MiB;
    unified_cache *  cache         = get_unified_cache_for_device(0);
    const size_t     pinned_before = host_ledger_sample().pinned_bytes;
    const uint64_t   pool_before   = cache != nullptr ? cache->pinned_pool_committed() : 0;
    if (cache == nullptr || cache->pre_allocate_runtime_chunks(requested) == 0) {
        void_arm(r, "the pinned commit between the stage runs was not made");
    }
    const uint64_t           injected     = cache != nullptr ? cache->pinned_pool_committed() - pool_before : 0;
    const host_budget_sample sample_after = host_ledger_sample();
    host_budget_sample       sample_prior = sample_after;
    sample_prior.pinned_bytes             = pinned_before;
    const size_t committed_now            = host_ledger_committed();
    const size_t prior                    = host_ledger_compute_budget(sample_prior, committed_now);
    const size_t fresh                    = host_ledger_compute_budget(sample_after, committed_now);
    if (injected == 0 || sample_after.pinned_bytes != pinned_before + injected) {
        void_arm(r, "the pinned share moved from " + num(pinned_before) + " to " + num(sample_after.pinned_bytes) +
                        " bytes, the pool committed " + num(injected));
    }
    if (prior != fresh + injected) {
        void_arm(r, "the pinned share is not the binding term: a commit of " + num(injected) +
                        " bytes moved the budget from " + num(prior) + " to " + num(fresh));
    }
    if (prior != S1) {
        void_arm(r, "the budget before the commit is " + num(prior) + ", not the stored " + num(S1) +
                        ", so a second store would not show as the stored value less the commit");
    }

    log_clear();
    const ggml_sycl_lifecycle_result late_rc = ggml_backend_sycl_stage_inventory_plan(&inv.inv, nullptr, false);
    const std::vector<size_t>        late    = plan_budgets();
    if (late_rc != GGML_SYCL_LIFECYCLE_OK || late.size() < n_dev) {
        void_arm(r, "the late stage run returned " + num(late_rc) + " with " + num(late.size()) +
                        " plans, want one per exposed device (" + num(n_dev) + ")");
    }
    expect(r, stores() == s0 + 1, "the transaction stored " + num(stores() - s0) + " times across its runs, want 1");
    expect(r, all_equal(late, S1), "the late plans recorded " + list(late) + ", not the early value " + num(S1));
    (void) ggml_backend_sycl_model_load_end(txn, false, nullptr);

    // a second transaction stores again
    ggml_sycl_load_txn txn2{};
    if (ggml_backend_sycl_model_load_begin(&txn2) != GGML_SYCL_LIFECYCLE_OK) {
        void_arm(r, "the second load transaction did not begin");
        return r;
    }
    s0 = stores();
    (void) ggml_backend_sycl_stage_inventory_plan(&inv.inv, nullptr, true);
    expect(r, stores() == s0 + 1, "the second transaction stored " + num(stores() - s0) + " times, want 1");
    (void) ggml_backend_sycl_model_load_end(txn2, false, nullptr);

    // the planner outside any scope reads B0 and stores nothing
    ggml_sycl_test_host_ledger_store_budget(B0);
    s0 = stores();
    placement_kv_info    kv{};
    const placement_plan bare = compute_placement_plan(planner_inventory(), 1 * MiB, 0, kv, nullptr, k_experts);
    expect(r, bare.host_budget == B0, "the planner outside a scope recorded " + num(bare.host_budget) + ", not B0");
    expect(r, stores() == s0 && host_ledger_budget() == B0, "the planner outside a scope stored a budget");
    return r;
}

const char * outcome_name(outcome o) {
    return o == outcome::PASS ? "PASS" : o == outcome::FAIL ? "FAIL" : "VOID";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--strict-child") == 0) {
        return strict_child(argv[2]);
    }
    if (!sycl_test_require_gpu("the host ledger's device arms")) {
        return SYCL_TEST_SKIP;
    }
    rlimit lim{};
    if (getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY && lim.rlim_cur < 64 * MiB) {
        std::printf("SKIP: RLIMIT_MEMLOCK is %llu bytes; the HOST_CPU arms lock several mappings\n",
                    (unsigned long long) lim.rlim_cur);
        return SYCL_TEST_SKIP;
    }
    // Small pool chunks keep the zone commits in MiB, and a 5 % host budget makes the percentage term the binding one,
    // so the budget moves by exactly a pinned commit (arm (j)). Both are read when the caches are built.
    setenv("GGML_SYCL_PINNED_CHUNK_MB", "16", 1);
    ggml_backend_sycl_set_unified_cache_host_budget_pct(5);
    g_test_thread = std::this_thread::get_id();
    ggml_log_set(capture_log, nullptr);

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b0(ggml_backend_sycl_init(0), ggml_backend_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> b1(
        ggml_backend_sycl_get_device_count() > 1 ? ggml_backend_sycl_init(1) : nullptr, ggml_backend_free);
    if (!b0) {
        std::printf("FAIL: backend 0 did not initialise\n");
        return 1;
    }

    struct named_arm {
        const char * name;
        arm_result (*run)();
    };

    const named_arm arms[] = {
        { "(l) class C never yields",                                  arm_l_class_c_never_yields          },
        { "(n) a yield excludes its own load",                         arm_n_yield_excludes_own_load       },
        { "(k) every owner exit accounts",                             arm_k_every_exit_accounts           },
        { "(m) the must_host_cpu request rules",                       arm_m_request_rules                 },
        { "(o) the cohort and the tier backstop",                      arm_o_cohort_and_backstop           },
        { "over budget: one would-refuse WARN, and the load proceeds", fixture_over_budget_warns_and_loads },
        { "plan against committed, on the pool",                       fixture_plan_against_committed      },
        { "(j) the store order",                                       arm_j_store_order                   },
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
    std::printf("host-ledger device arms: %d passed, %d failed, %d void\n", passed, failed, voided);
    return failed == 0 && voided == 0 ? 0 : 1;
}
