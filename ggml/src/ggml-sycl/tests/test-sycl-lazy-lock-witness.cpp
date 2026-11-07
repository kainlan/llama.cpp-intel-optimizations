//
// The five witnessed locks (th32 §3.3, §11 T5; canonical contract §12.5): the IO pool's reap, a yield round and
// host_ledger_sample() each run with none of them held. The reap's positive control takes each lock in turn and calls
// a FULL reap under it, which must refuse naming that lock; a table's strand lock is not in this tree, and the control
// says so. A yield round and host_ledger_sample() each abort under every lock, through GGML_SYCL_WITNESS, and return
// with none held, which is the negative control that a witness which always fires is not mistaken for one that works.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN. Every case runs in a child of this binary, so a reap that takes the registry's mutex under itself, or an
// abort, ends only that child: a child bounded at 5 s that does not return is VOID, which fails the binary like a
// FAIL. The children of the queue lock initialise backend 0 and ensure a block ring, so that an IO engine and its
// queue lock exist; the binary is registered gpu-serial, serial, with the selector pinned to the B50
// (level_zero:1). The parent initialises no backend.

#include "../lazy-io-pool.hpp"
#include "../unified-cache.hpp"
#include "../witnessed-mutex.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-test-harness.hpp"
#include "sycl-test-skip.hpp"

#include <signal.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ggml_sycl;
using namespace host_ledger_test_harness;
using host_ledger_test_harness::arm_loop::named_arm;
using host_ledger_test_harness::arm_loop::run_arms;

namespace {

constexpr uint64_t GiB = 1024ull * 1024ull * 1024ull;

// The bound of §11 T5 for a child that initialises nothing, and the bound for a queue-lock child, which initialises
// backend 0 and ensures a ring before it takes the lock: the lock's own wait is the 5 s one inside it.
constexpr std::chrono::milliseconds k_bound{ 5000 };
constexpr std::chrono::milliseconds k_backend_bound{ 120000 };

// The queue-lock children's ring: LAZY_BLOCK_RING_QD slots of 8192 B, created through the coverage pin on device 0
// against a stored budget far above it.
constexpr uint64_t k_need   = LAZY_BLOCK_RING_QD * 8192;
constexpr uint64_t k_budget = 64 * GiB;

const char * k_self         = "test-sycl-lazy-lock-witness";
const char * k_reap_child   = "--reap-child";
const char * k_ledger_child = "--ledger-child";

// The five locks, by lazy_witnessed_lock, with the name each one's witness gives it.
struct lock_entry {
    int          lock;
    const char * name;
};

const lock_entry k_locks[] = {
    { LAZY_LOCK_REGISTRY,  "g_lazy_registry_mutex"    },
    { LAZY_LOCK_INVENTORY, "g_tensor_inventory_mutex" },
    { LAZY_LOCK_PLAN,      "g_lifecycle_plan_mutex"   },
    { LAZY_LOCK_QUEUE,     "lazy_io_engine::mutex"    },
    { LAZY_LOCK_STRAND,    "a table's strand lock"    },
};

// The two entries that must run with no witnessed lock held (§3.3, §4.2), each with the text its witness aborts with.
struct ledger_entry {
    const char * what;   // the child's argument
    const char * label;  // the arm's wording
    const char * abort;  // GGML_SYCL_WITNESS's message
};

const ledger_entry k_sample = {
    "sample", "host_ledger_sample",
    "[HOST-LEDGER] host_ledger_sample ran holding a lazy table's witnessed lock (th32 3.3)"
};
const ledger_entry k_yield = { "yield", "a yield round",
                               "[HOST-LEDGER] a yield round ran holding a lazy table's witnessed lock (th32 3.3)" };

const char * k_arm_reap[] = {
    "a FULL reap under the IO pool registry's leaf mutex refuses naming it",
    "a FULL reap under g_tensor_inventory_mutex refuses naming it",
    "a FULL reap under g_lifecycle_plan_mutex refuses naming it",
    "a FULL reap under the IO engine's queue lock refuses naming it",
    "the reap's control reports a table's strand lock not wired",
};
const char * k_arm_sample = "host_ledger_sample aborts under each witnessed lock and returns with none held";
const char * k_arm_yield  = "a yield round aborts under each witnessed lock and returns with none held";

// --- small readings ---

std::string status_text(int status) {
    if (WIFSIGNALED(status)) {
        return "signal " + std::to_string(WTERMSIG(status));
    }
    return "exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

bool exited_zero(const child_run & run) {
    return WIFEXITED(run.status) && WEXITSTATUS(run.status) == 0;
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

std::vector<std::string> child_env() {
    // GGML_SYCL_WITNESS is on by default in this PRIVATE_TESTING build; the setting states it, so an inherited 0
    // cannot turn the death children into returns.
    return child_environment({ "GGML_SYCL_WITNESS_CHECKS=1", "GGML_NO_BACKTRACE=1" });
}

std::chrono::milliseconds bound_for(int lock) {
    return lock == LAZY_LOCK_QUEUE ? k_backend_bound : k_bound;
}

// --- the children ---

// A child's way out: its line, then _Exit, so no teardown runs in a process that may still hold a lock.
[[noreturn]] void child_exit(const char * tag, const std::string & line, int code) {
    std::fprintf(stderr, "%s %s\n", tag, line.c_str());
    std::fflush(stderr);
    std::_Exit(code);
}

// The queue lock exists only with a live IO engine, which a ring's ensure starts: backend 0 builds the cache the ring
// is charged to, and the pin's ensure creates it. The pin is leaked on purpose, so the engine outlives the call.
void live_engine(const char * tag) {
    ggml_sycl_test_host_ledger_store_budget(k_budget);
    if (ggml_backend_sycl_init(0) == nullptr) {
        child_exit(tag, "void: backend 0 did not initialise", SYCL_TEST_SKIP);
    }
    (void) new lazy_test_coverage_pin(ggml_sycl_test_make_coverage_pin(k_need));
}

int reap_child(int lock) {
    if (lock == LAZY_LOCK_QUEUE) {
        live_engine("REAP-HOLDING");
    }
    const lazy_reap_holding_result res = ggml_sycl_test_lazy_ring_reap_holding(lock);
    char                           line[256];
    std::snprintf(line, sizeof(line), "status=%s witnessed=%d registry_held=%d refused=%d held=%s",
                  lazy_holding_status_name(res.status), res.witnessed ? 1 : 0, res.registry_held ? 1 : 0,
                  res.refused ? 1 : 0, res.held != nullptr ? res.held : "(none)");
    child_exit("REAP-HOLDING", line, 0);
}

void call_sample(void *) {
    (void) host_ledger_sample();
}

void call_yield(void *) {
    (void) host_ledger_test_yield_round(1, 0);
}

int ledger_child(const char * what, const char * lock_arg) {
    const lazy_hold_fn fn = std::strcmp(what, k_sample.what) == 0 ? call_sample : call_yield;
    if (std::strcmp(lock_arg, "none") == 0) {
        fn(nullptr);
        child_exit("LEDGER-CHILD", "returned holding none", 0);
    }
    const int lock = std::atoi(lock_arg);
    if (lock == LAZY_LOCK_QUEUE) {
        live_engine("LEDGER-CHILD");
    }
    const lazy_holding_status status = ggml_sycl_test_lazy_run_holding_any(lock, fn, nullptr);
    child_exit("LEDGER-CHILD", std::string("returned status=") + lazy_holding_status_name(status), 0);
}

// --- the reap's positive control ---

// One lock's run of the control (§11 T5): a FULL reap called on a thread that holds the lock reads it through the
// witness, prints one "[HOST-PLAN-BUG] ring reap refused: <name> held" line, reaps nothing and returns;
// registry_mutex_held() reads the registry's own mutex held and no other. A table's strand lock reports not-wired and
// no line. A child that does not return within its bound is VOID: under the registry's mutex a reap that takes that
// mutex again never returns.
// Engagement, VOID counting as failed: the child returned, and for the queue lock found a live IO engine.
arm_result reap_under(int lock) {
    arm_result         r;
    const lock_entry & e = k_locks[lock];
    const child_run    run =
        run_child_strings({ k_self, k_reap_child, std::to_string(lock) }, child_env(), bound_for(lock));
    if (!run.ran) {
        void_arm(r, std::string("the reap under ") + e.name + " did not return within " +
                        num(bound_for(lock).count() / 1000) + " s");
        return r;
    }
    const std::string got = first_line_with(run.err, "REAP-HOLDING");
    if (got.find("REAP-HOLDING void:") != std::string::npos || got.find("status=no-engine") != std::string::npos) {
        void_arm(r, std::string("the reap under ") + e.name + " did not engage: " + got);
        return r;
    }
    const bool        wired = lock != LAZY_LOCK_STRAND;
    const std::string want =
        wired ? std::string("REAP-HOLDING status=held witnessed=1 registry_held=") +
                    (lock == LAZY_LOCK_REGISTRY ? "1" : "0") + " refused=1 held=" + e.name :
                std::string("REAP-HOLDING status=not-wired witnessed=0 registry_held=0 refused=0 held=(none)");
    const std::string warn    = std::string("[HOST-PLAN-BUG] ring reap refused: ") + e.name + " held";
    const size_t      refusal = text_count_with(run.err, "[HOST-PLAN-BUG] ring reap refused");
    expect(r, exited_zero(run) && text_count_with(run.err, want.c_str()) == 1,
           std::string("the reap under ") + e.name + " ended with " + status_text(run.status) + " printing '" + got +
               "', want exit 0 and '" + want + "'");
    expect(r, wired ? text_count_with(run.err, warn.c_str()) == 1 && refusal == 1 : refusal == 0,
           std::string("the reap under ") + e.name + " printed " + num(refusal) + " refusal line(s), want " +
               (wired ? "1 '" + warn + "'" : std::string("0")));
    return r;
}

template <int L> arm_result arm_reap_under() {
    return reap_under(L);
}

// --- the ledger's two entries ---

// One entry under every lock (§3.3, §4.2): with no lock held the child returns, and under each of the four witnessed
// locks of this tree it aborts through GGML_SYCL_WITNESS with the entry's message, before it returns.
// Engagement, VOID counting as failed: the control without a lock returned, and every child ended within its bound;
// the queue lock's child found a live IO engine.
arm_result ledger_arm(const ledger_entry & e) {
    arm_result      r;
    const child_run control = run_child_strings({ k_self, k_ledger_child, e.what, "none" }, child_env(), k_bound);
    if (!control.ran || !exited_zero(control) ||
        text_count_with(control.err, "LEDGER-CHILD returned holding none") != 1 ||
        text_count_with(control.err, e.abort) != 0) {
        void_arm(r, std::string(e.label) + " with no lock held " +
                        (control.ran ? "ended with " + status_text(control.status) : std::string("did not return")) +
                        " printing '" + first_line_with(control.err, "LEDGER-CHILD") + "', want exit 0 and a return");
        return r;
    }
    for (const lock_entry & l : k_locks) {
        if (l.lock == LAZY_LOCK_STRAND) {
            continue;
        }
        const child_run run = run_child_strings({ k_self, k_ledger_child, e.what, std::to_string(l.lock) }, child_env(),
                                                bound_for(l.lock));
        if (!run.ran) {
            void_arm(r, std::string(e.label) + " under " + l.name + " did not end within " +
                            num(bound_for(l.lock).count() / 1000) + " s");
            continue;
        }
        if (text_count_with(run.err, "LEDGER-CHILD void:") != 0 || text_count_with(run.err, "status=no-engine") != 0) {
            void_arm(r, std::string(e.label) + " under " + l.name +
                            " did not engage: " + first_line_with(run.err, "LEDGER-CHILD"));
            continue;
        }
        const bool died = WIFSIGNALED(run.status) && WTERMSIG(run.status) == SIGABRT;
        expect(r,
               died && text_count_with(run.err, e.abort) == 1 && text_count_with(run.err, "LEDGER-CHILD returned") == 0,
               std::string(e.label) + " under " + l.name + " ended its child with " + status_text(run.status) +
                   " after " + num(text_count_with(run.err, "LEDGER-CHILD returned")) + " return line(s) and " +
                   num(text_count_with(run.err, e.abort)) + " line(s) '" + e.abort + "', want SIGABRT, 0 and 1");
    }
    return r;
}

arm_result arm_sample() {
    return ledger_arm(k_sample);
}

arm_result arm_yield() {
    return ledger_arm(k_yield);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], k_reap_child) == 0) {
        return reap_child(std::atoi(argv[2]));
    }
    if (argc == 4 && std::strcmp(argv[1], k_ledger_child) == 0) {
        return ledger_child(argv[2], argv[3]);
    }
    if (argc != 1) {
        std::printf("FAIL: unknown arguments; this binary takes none\n");
        return 1;
    }
    // The queue lock's children need a device; without one every queue case would VOID, so the binary skips.
    if (!sycl_test_require_gpu("the lazy lock witness arms")) {
        return SYCL_TEST_SKIP;
    }
    const std::vector<named_arm> arms = {
        { k_arm_reap[LAZY_LOCK_REGISTRY],  arm_reap_under<LAZY_LOCK_REGISTRY>  },
        { k_arm_reap[LAZY_LOCK_INVENTORY], arm_reap_under<LAZY_LOCK_INVENTORY> },
        { k_arm_reap[LAZY_LOCK_PLAN],      arm_reap_under<LAZY_LOCK_PLAN>      },
        { k_arm_reap[LAZY_LOCK_QUEUE],     arm_reap_under<LAZY_LOCK_QUEUE>     },
        { k_arm_reap[LAZY_LOCK_STRAND],    arm_reap_under<LAZY_LOCK_STRAND>    },
        { k_arm_sample,                    arm_sample                          },
        { k_arm_yield,                     arm_yield                           },
    };
    return run_arms(arms, "lazy lock witness arms");
}
