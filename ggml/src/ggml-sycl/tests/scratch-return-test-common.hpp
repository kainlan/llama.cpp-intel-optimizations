//
// What the SCRATCH return's three device REDs share (docs/design/sycl-mmap-host-tier.md §7 C4:
// test-sycl-scratch-trial-return, test-sycl-configure-refusal, test-sycl-scratch-owner-end): the entry checks, the
// mutant selector's names, the case runner and the pool readings every case states its figures from.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Every case runs in its own child process, re-executed from /proc/self/exe with the driver's own environment (it
// carries the registration's ONEAPI_DEVICE_SELECTOR, LD_LIBRARY_PATH and GGML_SYCL_PINNED_CHUNK_MB). So each case
// starts from a fresh cache, a fresh ledger and no transaction, a case that aborts cannot take the others with it, and
// a planted fault that leaves the pool unable to shut down still lets its case print the line that names it. The
// driver relays each child's stderr, so a mutant's ctest reads the child's failure line in the driver's output.
//
// A case's verdict is its exit status: 0 PASS, 1 FAIL, 77 VOID (its entry or engagement check found it could not
// reach the mechanism it scores). A case that dies by a signal is a FAIL. The driver exits 1 when any case FAILed,
// else 77 when any was VOID, else 0, so a VOID never reads as a pass.

#pragma once

#include "../pinned-pool.hpp"
#include "../scratch-test-mutant.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "host-ledger-test-harness.hpp"
#include "sycl-test-skip.hpp"

#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern char ** environ;

namespace scratch_return_test {

using namespace ggml_sycl;
using namespace host_ledger_test_harness;

constexpr size_t MiB = 1024ull * 1024ull;

// The registration's chunk size (GGML_SYCL_PINNED_CHUNK_MB=64): every figure below is a sum of 64 MiB terms.
constexpr size_t k_chunk_mib = 64;

// --- the mutant selector's names ---

struct mutant_name {
    const char *        name;
    scratch_test_mutant mutant;
};

inline const std::vector<mutant_name> & mutant_names() {
    static const std::vector<mutant_name> names = {
        { "double-decrement",          scratch_test_mutant::DOUBLE_DECREMENT          },
        { "footprint-before-free",     scratch_test_mutant::FOOTPRINT_BEFORE_FREE     },
        { "carrier-dtor-noop",         scratch_test_mutant::CARRIER_DTOR_NOOP         },
        { "moved-from-releases",       scratch_test_mutant::MOVED_FROM_RELEASES       },
        { "no-prune",                  scratch_test_mutant::NO_PRUNE                  },
        { "prune-without-query",       scratch_test_mutant::PRUNE_WITHOUT_QUERY       },
        { "subtract-at-refusal",       scratch_test_mutant::SUBTRACT_AT_REFUSAL       },
        { "dtor-no-report",            scratch_test_mutant::DTOR_NO_REPORT            },
        { "stranded-reserve-8",        scratch_test_mutant::STRANDED_RESERVE_8        },
        { "lease-checked-first",       scratch_test_mutant::LEASE_CHECKED_FIRST       },
        { "withdraw-transition-blind", scratch_test_mutant::WITHDRAW_TRANSITION_BLIND },
        { "configure-rebuilds",        scratch_test_mutant::CONFIGURE_REBUILDS        },
        { "carrier-assert-removed",    scratch_test_mutant::CARRIER_ASSERT_REMOVED    },
        { "registered-exact-ptr-only", scratch_test_mutant::REGISTERED_EXACT_PTR_ONLY },
        { "return-releases-live",      scratch_test_mutant::RETURN_RELEASES_LIVE      },
        { "return-no-withdraw",        scratch_test_mutant::RETURN_NO_WITHDRAW        },
        { "return-no-file",            scratch_test_mutant::RETURN_NO_FILE            },
        { "return-no-immediate",       scratch_test_mutant::RETURN_NO_IMMEDIATE       },
        { "release-no-erase",          scratch_test_mutant::RELEASE_NO_ERASE          },
        { "configure-direct-release",  scratch_test_mutant::CONFIGURE_DIRECT_RELEASE  },
        { "strict-abort-dropped",      scratch_test_mutant::STRICT_ABORT_DROPPED      },
        { "abort-non-strict",          scratch_test_mutant::ABORT_NON_STRICT          },
        { "orphan-free-ignored",       scratch_test_mutant::ORPHAN_FREE_IGNORED       },
        { "owner-end-no-empty",        scratch_test_mutant::OWNER_END_NO_EMPTY        },
        { "owner-end-first-only",      scratch_test_mutant::OWNER_END_FIRST_ONLY      },
        { "lease-drop-no-index",       scratch_test_mutant::LEASE_DROP_NO_INDEX       },
        { "begin-names-any-owner",     scratch_test_mutant::BEGIN_NAMES_ANY_OWNER     },
        { "return-noop",               scratch_test_mutant::RETURN_NOOP               },
        { "rtm-left-in-txn",           scratch_test_mutant::RTM_LEFT_IN_TXN           },
        { "keep-walks-set",            scratch_test_mutant::KEEP_WALKS_SET            },
        { "owner-end-removed",         scratch_test_mutant::OWNER_END_REMOVED         },
    };
    return names;
}

// "a,b" to the selector's bits. False, with the unknown name, when a name is not in the table: a typo in a
// registration must not run the unmutated binary and read as a missed mutant.
inline bool mutants_parse(const std::string & list, uint64_t * bits, std::string * unknown) {
    *bits     = 0;
    size_t at = 0;
    while (at <= list.size() && !list.empty()) {
        const size_t      comma = list.find(',', at);
        const std::string name  = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        bool              found = false;
        for (const mutant_name & m : mutant_names()) {
            if (name == m.name) {
                *bits |= 1ull << static_cast<int>(m.mutant);
                found = true;
            }
        }
        if (!found) {
            *unknown = name;
            return false;
        }
        if (comma == std::string::npos) {
            break;
        }
        at = comma + 1;
    }
    return true;
}

// --- printing ---

inline std::string mib(size_t bytes) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f", bytes / static_cast<double>(MiB));
    return buf;
}

// Every failing check prints its own line and the case runs on, so one fault never hides the next (§7 C4 r12 N-4).
struct checks {
    const char * name;
    bool         failed = false;
    bool         voided = false;

    void fail(const std::string & why) {
        failed = true;
        fprintf(stderr, "FAIL [%s] %s\n", name, why.c_str());
    }

    void expect(bool ok, const std::string & why) {
        if (!ok) {
            fail(why);
        }
    }

    // The case could not reach its mechanism: the figures it would score mean nothing.
    void void_case(const std::string & why) {
        voided = true;
        fprintf(stderr, "VOID [%s] %s\n", name, why.c_str());
    }

    int status() const {
        if (failed) {
            return 1;
        }
        return voided ? SYCL_TEST_SKIP : 0;
    }
};

// --- the entry checks ---

// The registration pins the device and the chunk size; a run without either measures something else.
inline bool entry_environment_ok(checks & c) {
    const char * selector = getenv("ONEAPI_DEVICE_SELECTOR");
    if (selector == nullptr || selector[0] == '\0') {
        c.void_case("ONEAPI_DEVICE_SELECTOR is unset: the registration pins one device (§M88 (2))");
        return false;
    }
    const char * chunk_mb = getenv("GGML_SYCL_PINNED_CHUNK_MB");
    if (chunk_mb == nullptr || std::strtoull(chunk_mb, nullptr, 10) != k_chunk_mib) {
        c.void_case("GGML_SYCL_PINNED_CHUNK_MB is not 64: every expected figure is a sum of 64 MiB chunk terms");
        return false;
    }
    if (sycl_test_gpu_devices().empty()) {
        sycl_test_print_skip("the SCRATCH return");
        c.void_case("no SYCL GPU device");
        return false;
    }
    return true;
}

// --- the device's cache and pool ---

struct device_fixture {
    ggml_backend_t      backend = nullptr;  // keeps the device initialised; its cache lives until module shutdown
    unified_cache *     cache   = nullptr;
    pinned_chunk_pool * pool    = nullptr;
};

// A backend on device 0 (the selector's one device), its cache with host zones configured so SCRATCH is 64 MiB, and
// that cache's pool. VOID when the cache already had host zones (the figures assume a fresh layout) or has no pool.
inline bool device_fixture_open(checks & c, device_fixture & f) {
    f.backend = ggml_backend_sycl_init(0);
    if (f.backend == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return false;
    }
    f.cache = get_unified_cache_for_device(0);
    if (f.cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return false;
    }
    f.pool = f.cache->host_arena_for_testing();
    if (f.pool == nullptr) {
        c.void_case("device 0's cache has no pinned pool");
        return false;
    }
    if (f.cache->host_zones_configured()) {
        c.void_case("device 0's host zones were configured before the fixture");
        return false;
    }
    if (f.pool->chunk_size() != k_chunk_mib * MiB) {
        c.void_case("the pool's chunk size is " + mib(f.pool->chunk_size()) + " MiB, not 64");
        return false;
    }
    // SCRATCH 64 MiB on top of the smallest other zones the pool lays out.
    if (!f.cache->configure_host_zones(MiB, MiB, MiB, k_chunk_mib * MiB) || !f.cache->host_zones_configured()) {
        c.void_case("configure_host_zones(1, 1, 1, 64 MiB) left the zones unconfigured");
        return false;
    }
    if (f.cache->host_zone_capacity(host_zone_id::SCRATCH) != k_chunk_mib * MiB) {
        c.void_case("SCRATCH capacity is " + mib(f.cache->host_zone_capacity(host_zone_id::SCRATCH)) +
                    " MiB after configure, not 64.0");
        return false;
    }
    return true;
}

// Fills SCRATCH's existing free blocks, so the next allocation of any size lands in a chunk grown after this call.
// The fillers stay live for the whole case.
inline std::vector<void *> scratch_fill(unified_cache * cache) {
    std::vector<void *> fillers;
    for (;;) {
        const size_t block = cache->host_zone_largest_free_block(host_zone_id::SCRATCH);
        if (block < MiB) {
            break;
        }
        void * p = cache->host_zone_alloc(host_zone_id::SCRATCH, block - pinned_chunk_pool::DEFAULT_ALIGNMENT,
                                          pinned_chunk_pool::DEFAULT_ALIGNMENT);
        if (p == nullptr) {
            break;
        }
        fillers.push_back(p);
    }
    return fillers;
}

// The chunk index of a pointer into a base chunk (its handle is the index), or SIZE_MAX.
inline size_t chunk_of(const pinned_chunk_pool * pool, const void * p) {
    const uint64_t handle = pool->find_chunk_handle(p);
    if (handle == pinned_chunk_pool::INVALID_CHUNK_HANDLE || (handle >> 63) != 0) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(handle);
}

// How many captured log lines contain `a` (and `b`).
inline size_t lines(const char * a, const char * b = nullptr) {
    return log_count_with(a, b);
}

inline size_t lines(const char * a, const std::string & b) {
    return log_count_with(a, b.c_str());
}

// The cache's two release lines (unified-cache.cpp print_marked_released, print_marked_refusals).
constexpr const char * k_released_line = "[HOST-ARENA] SCRATCH released";
constexpr const char * k_refused_line  = "[HOST-ARENA] SCRATCH return refused";

// Fills SCRATCH until every free block is below `limit`, leaving each filled block with limit / 2 free. The fillers
// stay live until the case frees them.
inline std::vector<void *> scratch_fill_below(unified_cache * cache, size_t limit) {
    std::vector<void *> fillers;
    for (;;) {
        const size_t block = cache->host_zone_largest_free_block(host_zone_id::SCRATCH);
        if (block < limit) {
            break;
        }
        void * p =
            cache->host_zone_alloc(host_zone_id::SCRATCH, block - limit / 2, pinned_chunk_pool::DEFAULT_ALIGNMENT);
        if (p == nullptr) {
            break;
        }
        fillers.push_back(p);
    }
    return fillers;
}

inline void scratch_free_all(unified_cache * cache, std::vector<void *> & ptrs) {
    for (void * p : ptrs) {
        cache->host_zone_free(host_zone_id::SCRATCH, p);
    }
    ptrs.clear();
}

// The highest chunk index the SCRATCH zone holds state for: a grown chunk is appended, so after a grow it is the
// chunk that grow made. Scanned, because the pool keeps tombstones in place and exposes no slot count; the fixtures
// stay far below the bound. SIZE_MAX when the zone holds none.
inline size_t newest_scratch_chunk(const pinned_chunk_pool * pool) {
    size_t newest = SIZE_MAX;
    for (size_t i = 0; i < 1024; ++i) {
        if (pool->zone_chunk_usage(host_zone_id::SCRATCH, i).found) {
            newest = i;
        }
    }
    return newest;
}

// Right after RETURN: the chunk is withdrawn and filed in the marked set once.
inline void expect_returned(checks & c, const device_fixture & f, size_t idx, const char * when = "after RETURN") {
    const pinned_chunk_usage u = f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx);
    c.expect(u.withdrawn, "chunk " + std::to_string(idx) + " not withdrawn " + when);
    const size_t filed = f.cache->marked_entries(idx);
    c.expect(filed == 1, "chunk " + std::to_string(idx) + " filed " + std::to_string(filed) +
                             " times in the marked set " + when + ", expected 1");
}

// A release leaves no marked-set entry (§7 C4 r11 m-5): the set is empty, and a following begin of `owner` prints
// nothing and moves no chunk.
inline void expect_marked_set_quiet(checks & c, const device_fixture & f, size_t idx, uint64_t owner) {
    const size_t entries = f.cache->marked_entries();
    c.expect(entries == 0, "marked set holds " + std::to_string(entries) + " entry(ies) after the release of chunk " +
                               std::to_string(idx) + ", expected 0");
    const size_t printed = lines(k_released_line) + lines(k_refused_line);
    const size_t chunks  = f.pool->chunk_count();
    (void) f.cache->host_scratch_begin_release(owner);
    c.expect(lines(k_released_line) + lines(k_refused_line) == printed && f.pool->chunk_count() == chunks,
             "a begin after the release of chunk " + std::to_string(idx) + " printed or moved something");
}

// --- the driver ---

// The cases a run selects: all of them, or the comma-separated `--cases` list, which a mutant's registration names so
// its arm runs only the cases that score its fault. False, with the unknown name, when the list names no case.
inline bool cases_select(const std::vector<const char *> & all,
                         const std::string &               list,
                         std::vector<const char *> *       out,
                         std::string *                     unknown) {
    out->clear();
    if (list.empty()) {
        *out = all;
        return true;
    }
    size_t at = 0;
    for (;;) {
        const size_t      comma = list.find(',', at);
        const std::string name  = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        bool              found = false;
        for (const char * c : all) {
            if (name == c) {
                out->push_back(c);
                found = true;
            }
        }
        if (!found) {
            *unknown = name;
            return false;
        }
        if (comma == std::string::npos) {
            return true;
        }
        at = comma + 1;
    }
}

// Runs `cases` one per child, each as `<exe> --case <name> [--mutant <list>]`, with `envp` (the driver's environment
// when null). Returns the driver's exit status.
inline int run_cases(const char *                      exe,
                     const std::vector<const char *> & cases,
                     const std::string &               mutants,
                     char * const *                    envp,
                     std::chrono::milliseconds         bound) {
    bool any_fail = false;
    bool any_void = false;
    for (const char * name : cases) {
        std::vector<std::string> args = { exe, "--case", name };
        if (!mutants.empty()) {
            args.push_back("--mutant");
            args.push_back(mutants);
        }
        std::vector<char *> argv;
        for (std::string & a : args) {
            argv.push_back(a.data());
        }
        argv.push_back(nullptr);
        const child_run r = run_child(argv.data(), envp ? envp : environ, bound);
        fputs(r.err.c_str(), stderr);
        if (!r.ran) {
            fprintf(stderr, "FAIL [%s] the case did not finish within %lld ms\n", name, (long long) bound.count());
            any_fail = true;
        } else if (WIFSIGNALED(r.status)) {
            fprintf(stderr, "FAIL [%s] the case died by signal %d\n", name, WTERMSIG(r.status));
            any_fail = true;
        } else if (WEXITSTATUS(r.status) == SYCL_TEST_SKIP) {
            fprintf(stderr, "VOID [%s] (exit 77)\n", name);
            any_void = true;
        } else if (WEXITSTATUS(r.status) != 0) {
            fprintf(stderr, "FAIL [%s] exit %d\n", name, WEXITSTATUS(r.status));
            any_fail = true;
        } else {
            fprintf(stderr, "PASS [%s]\n", name);
        }
    }
    if (any_fail) {
        return 1;
    }
    return any_void ? SYCL_TEST_SKIP : 0;
}

// The arguments: a child's case name, the mutant list, and the driver's case selection.
struct child_args {
    std::string case_name;
    std::string mutants;
    std::string cases;
};

inline child_args child_args_parse(int argc, char ** argv) {
    child_args a;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0) {
            a.case_name = argv[++i];
        } else if (std::strcmp(argv[i], "--mutant") == 0) {
            a.mutants = argv[++i];
        } else if (std::strcmp(argv[i], "--cases") == 0) {
            a.cases = argv[++i];
        }
    }
    return a;
}

// The child's set-up: no core dump, the log capture and the selector. Returns false (exit 2) on an unknown mutant
// name. A core-dumping abort holds pinned memory past its exit message, and several planted faults end in an abort.
inline bool child_begin(const child_args & a) {
    const struct rlimit no_core = { 0, 0 };
    (void) setrlimit(RLIMIT_CORE, &no_core);
    ggml_log_set(capture_log, nullptr);
    uint64_t    bits = 0;
    std::string unknown;
    if (!mutants_parse(a.mutants, &bits, &unknown)) {
        fprintf(stderr, "unknown mutant '%s'\n", unknown.c_str());
        return false;
    }
    scratch_test_mutants_set(bits);
    return true;
}

// A test's main: a child runs its one case through `run_case`; the driver runs the selected cases, one child each.
inline int cases_main(int                               argc,
                      char **                           argv,
                      const std::vector<const char *> & all,
                      int (*run_case)(const child_args &),
                      std::chrono::milliseconds bound) {
    const child_args a = child_args_parse(argc, argv);
    if (!a.case_name.empty()) {
        return run_case(a);
    }
    std::vector<const char *> selected;
    std::string               unknown;
    if (!cases_select(all, a.cases, &selected, &unknown)) {
        fprintf(stderr, "unknown case '%s'\n", unknown.c_str());
        return 2;
    }
    // The driver touches no device: each case runs in its own child.
    return run_cases(argv[0], selected, a.mutants, nullptr, bound);
}

}  // namespace scratch_return_test
