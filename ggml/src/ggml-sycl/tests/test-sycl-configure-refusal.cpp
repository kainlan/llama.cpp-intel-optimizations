//
// configure_zones refuses while a withdrawn SCRATCH chunk survives, aborting under GGML_SYCL_STRICT_LEASES and
// returning false without it (docs/design/sycl-mmap-host-tier.md §7 C4 test-sycl-configure-refusal; §4.4
// "configure_zones refuses while a withdrawn chunk survives"; r11 m-6; r13 M-4, M-5; r14 M-8; r15 M-4; r17 M-5;
// §M81 (e)).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// LEAD-RUN: its children build a unified cache and a pinned pool whose chunks come from a real queue, so it is
// registered `sycl;cache;mem-handle` plus gpu-serial, serial, with the selector pinned to one discrete card. It loads
// no model.
//
// The driver never touches a device. It runs its own binary three times with fork and execve, giving each child its
// environment through the envp argument, built before the fork as a copy of the driver's environ (which carries the
// registration's ONEAPI_DEVICE_SELECTOR, LD_LIBRARY_PATH, GGML_SYCL_PINNED_CHUNK_MB and GGML_SYCL_HOST_RESERVE_MB):
//   strict   GGML_SYCL_STRICT_LEASES=1. Builds the withdrawn-chunk state and calls configure_host_zones, which must
//            abort: scored as SIGABRT plus the pool's `configure_zones refused: chunk` line on its stderr.
//   plain    the variable erased. The configure cases: the refusal changes nothing, a configure after the release
//            succeeds, configure's own release leaves no marked-set entry, and the shortfall control. Scored as exit 0.
//   carrier  the plain child's envp. A pool destroyed with a released_chunk outstanding must abort: SIGABRT plus the
//            `released_chunk outstanding at pool destruction` line.
// No child calls setenv: ggml_sycl_strict_enabled() latches the environment on first call, and the fixture's cache
// construction reaches it first. Each envp also sets GGML_NO_BACKTRACE=1, so an abort does not attach a debugger.
//
// A child whose entry check finds the strict variable set (plain, carrier) or unset (strict), the selector unset, or no
// device exits 77, and the driver then exits 77, so a VOID never reads as a pass.
//
// The withdrawn-chunk state is the runtime throw's (§7 C4): SCRATCH's base filled, the reserve's own transaction
// (MARK) grows one chunk through the notify, a 48 MiB stand-in lands in it, and RETURN withdraws it with the stand-in
// live. The plain child's shortfall control asks for more than the pool's budget, so the pool grows to that budget
// before it refuses; the registration bounds the budget with GGML_SYCL_HOST_RESERVE_MB, and the control is VOID above
// 2 GiB rather than commit the host's memory.
//
// Mutants (scratch-test-mutant.hpp), each a ctest of its own with its expected line; the registrations in
// ggml/src/ggml-sycl/CMakeLists.txt list them with the children each one runs.

#include "../common.hpp"
#include "scratch-return-test-common.hpp"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <string>
#include <vector>

using namespace scratch_return_test;

namespace {

constexpr const char * k_pool_refused_line = "configure_zones refused: chunk";
constexpr const char * k_carrier_line      = "released_chunk outstanding at pool destruction";

// The shortfall control's growth stops at the pool's budget; above this the control would commit the host's memory.
constexpr size_t k_shortfall_budget_max = 2048 * MiB;

// The sizes every configure call here asks for: they fit the base chunks once the withdrawn chunk is gone.
constexpr size_t k_scratch_new = 96 * MiB;

std::string num(size_t v) {
    return std::to_string(v);
}

bool strict_in_env() {
    const char * strict = getenv("GGML_SYCL_STRICT_LEASES");
    return strict != nullptr && std::atoi(strict) != 0;
}

uint64_t owner_of(ggml_backend_t backend) {
    return static_cast<const ggml_backend_sycl_context *>(backend->context)->host_zone_owner_id;
}

void exit_now(int status) {
    // No teardown: a fault that leaves the pool unable to shut down must not turn into a different verdict.
    fflush(stderr);
    _exit(status);
}

// --- the withdrawn-chunk state ---

struct withdrawn_state {
    std::vector<void *> fillers;
    void *              stand_in = nullptr;
    size_t              idx      = SIZE_MAX;
};

void state_free(const device_fixture & f, withdrawn_state & s) {
    if (s.stand_in != nullptr) {
        f.cache->host_zone_free(host_zone_id::SCRATCH, s.stand_in);
        s.stand_in = nullptr;
    }
    scratch_free_all(f.cache, s.fillers);
}

// The runtime throw's state. False after a VOID (the fixture could not reach it) or a FAIL (RETURN left the chunk
// counted, so no configure case can run over it); the caller frees what it holds.
bool state_build(checks & c, const device_fixture & f, withdrawn_state & s) {
    s.fillers       = scratch_fill(f.cache);
    const size_t n0 = f.pool->chunk_count();
    (void) ggml_backend_sycl_host_scratch_trial(f.backend, GGML_SYCL_HOST_SCRATCH_TRIAL_MARK, 512);
    (void) ggml_backend_sycl_notify_host_buffer_sizes(f.backend, 100 * MiB, 0);
    if (f.pool->chunk_count() != n0 + 1) {
        c.void_case("the 100 MiB notify grew " + num(f.pool->chunk_count() - n0) + " chunk(s), not 1");
        return false;
    }
    s.idx      = newest_scratch_chunk(f.pool);
    s.stand_in = f.cache->host_zone_alloc(host_zone_id::SCRATCH, 48 * MiB, pinned_chunk_pool::DEFAULT_ALIGNMENT);
    if (s.stand_in == nullptr || chunk_of(f.pool, s.stand_in) != s.idx) {
        c.void_case("the 48 MiB stand-in did not land in the chunk the notify grew");
        return false;
    }
    (void) ggml_backend_sycl_host_scratch_trial(f.backend, GGML_SYCL_HOST_SCRATCH_TRIAL_RETURN, 512);
    if (!f.pool->zone_chunk_usage(host_zone_id::SCRATCH, s.idx).withdrawn) {
        c.fail("chunk " + num(s.idx) + " not withdrawn after RETURN: the configure cases need the withdrawn state");
        return false;
    }
    return true;
}

// --- the children ---

int child_strict(checks & c, const device_fixture & f) {
    withdrawn_state s;
    if (!state_build(c, f, s)) {
        state_free(f, s);
        exit_now(c.status());
    }
    const bool ok = f.cache->configure_host_zones(MiB, MiB, MiB, k_scratch_new);
    c.fail(std::string("configure_host_zones returned ") + (ok ? "true" : "false") + " over withdrawn chunk " +
           num(s.idx) + " in the strict child");
    exit_now(1);
    return 1;
}

struct snapshot {
    size_t                capacity  = 0;
    size_t                chunks    = 0;
    bool                  withdrawn = false;
    size_t                marked    = 0;
    std::array<size_t, 4> reserved{};
};

snapshot snapshot_take(const device_fixture & f, size_t idx) {
    snapshot s;
    s.capacity  = f.cache->host_zone_capacity(host_zone_id::SCRATCH);
    s.chunks    = f.pool->chunk_count();
    s.withdrawn = f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).withdrawn;
    s.marked    = f.cache->marked_entries();
    for (size_t z = 0; z < s.reserved.size(); ++z) {
        s.reserved[z] = f.cache->host_zone_reserved_capacity(static_cast<host_zone_id>(z));
    }
    return s;
}

// The refusal over a withdrawn chunk changes nothing (r11 m-6; r13 M-5).
bool plain_refusal(checks & c, const device_fixture & f, withdrawn_state & s) {
    const snapshot    before  = snapshot_take(f, s.idx);
    const std::string named   = "chunk " + num(s.idx) + " still withdrawn";
    const size_t      refused = lines(k_pool_refused_line, named);
    const bool        ok      = f.cache->configure_host_zones(MiB, MiB, MiB, k_scratch_new);
    const snapshot    after   = snapshot_take(f, s.idx);
    if (ok) {
        c.fail("configure_zones rebuilt over withdrawn chunk " + num(s.idx) + ": capacity " + mib(after.capacity) +
               " after, " + mib(before.capacity) + " before, marked set holds " + num(after.marked));
        // The rebuild dropped the allocator under the live stand-in: nothing after it is a state to score or free.
        exit_now(1);
    }
    c.expect(lines(k_pool_refused_line, named) == refused + 1,
             "no `configure_zones refused: " + named + "` line for the refusal");
    c.expect(after.capacity == before.capacity, "SCRATCH capacity " + mib(after.capacity) + " MiB after the refusal, " +
                                                    mib(before.capacity) + " MiB before");
    c.expect(after.chunks == before.chunks,
             "chunk_count " + num(after.chunks) + " after the refusal, " + num(before.chunks) + " before");
    c.expect(after.withdrawn, "chunk " + num(s.idx) + " not withdrawn after the refusal");
    c.expect(after.marked == before.marked,
             "marked set holds " + num(after.marked) + " after the refusal, " + num(before.marked) + " before");
    for (size_t z = 0; z < before.reserved.size(); ++z) {
        c.expect(after.reserved[z] == before.reserved[z], "host zone " + num(z) + "'s reserved figure " +
                                                              mib(after.reserved[z]) + " MiB after the refusal, " +
                                                              mib(before.reserved[z]) + " MiB before");
    }
    return true;
}

// After the stand-in's free and a begin, the same call succeeds.
void plain_after_release(checks & c, const device_fixture & f, withdrawn_state & s) {
    const size_t idx = s.idx;
    state_free(f, s);
    const size_t released = lines(k_released_line);
    (void) f.cache->host_scratch_begin_release(owner_of(f.backend));
    c.expect(lines(k_released_line) > released, "no SCRATCH released line for chunk " + num(idx) + " at the begin");
    const bool ok = f.cache->configure_host_zones(MiB, MiB, MiB, k_scratch_new);
    c.expect(ok && f.cache->host_zones_configured(),
             "configure_host_zones did not configure the zones once chunk " + num(idx) + " was released");
    c.expect(f.cache->host_zone_capacity(host_zone_id::SCRATCH) == k_scratch_new,
             "SCRATCH capacity " + mib(f.cache->host_zone_capacity(host_zone_id::SCRATCH)) +
                 " MiB after the configure, expected " + mib(k_scratch_new) + " MiB");
    c.expect(f.cache->host_zone_reserved_capacity(host_zone_id::SCRATCH) == k_scratch_new,
             "SCRATCH's reserved figure " + mib(f.cache->host_zone_reserved_capacity(host_zone_id::SCRATCH)) +
                 " MiB after the configure, expected " + mib(k_scratch_new) + " MiB");
}

// configure's own release: a marked chunk already empty when configure runs is released by it (r13 M-4).
void plain_own_release(checks & c, const device_fixture & f) {
    withdrawn_state s;
    if (!state_build(c, f, s)) {
        state_free(f, s);
        return;
    }
    const size_t idx = s.idx;
    state_free(f, s);  // the chunk is empty and still marked: no begin runs
    const size_t released = lines(k_released_line);
    const bool   ok       = f.cache->configure_host_zones(MiB, MiB, MiB, k_scratch_new);
    c.expect(ok && f.cache->host_zones_configured(),
             "configure_host_zones did not configure the zones over empty marked chunk " + num(idx));
    c.expect(lines(k_released_line) > released, "no SCRATCH released line for chunk " + num(idx) + " at configure");
    c.expect(!f.pool->zone_chunk_usage(host_zone_id::SCRATCH, idx).found,
             "chunk " + num(idx) + " still in the SCRATCH zone after configure's own release");
    expect_marked_set_quiet(c, f, idx, owner_of(f.backend));
}

// The shortfall control (r12 M-6): sizes above what the pool can grow to return true, unconfigured, zeroed.
void plain_shortfall(checks & c, const device_fixture & f) {
    const size_t budget = f.pool->budget();
    if (budget > k_shortfall_budget_max) {
        c.void_case("the pool's budget is " + mib(budget) +
                    " MiB: the registration's GGML_SYCL_HOST_RESERVE_MB bounds the shortfall control's growth");
        return;
    }
    const bool ok = f.cache->configure_host_zones(MiB, MiB, MiB, budget + k_chunk_mib * MiB);
    c.expect(ok, "configure_host_zones returned false for sizes above the pool's budget");
    c.expect(!f.cache->host_zones_configured(), "the zones are configured after sizes above the pool's budget");
    for (size_t z = 0; z < 4; ++z) {
        const size_t reserved = f.cache->host_zone_reserved_capacity(static_cast<host_zone_id>(z));
        c.expect(reserved == 0, "host zone " + num(z) + "'s reserved figure " + mib(reserved) +
                                    " MiB after the shortfall, expected 0");
    }
}

int child_plain(checks & c, const device_fixture & f) {
    withdrawn_state s;
    if (!state_build(c, f, s)) {
        state_free(f, s);
        return c.status();
    }
    (void) plain_refusal(c, f, s);
    plain_after_release(c, f, s);
    plain_own_release(c, f);
    if (!c.failed) {
        plain_shortfall(c, f);
    }
    return c.status();
}

int child_carrier(checks & c, const device_fixture & f) {
    // A pool of its own on the cache's queue, so the child can destroy it.
    auto * pool = new pinned_chunk_pool(f.cache->get_queue(), 4 * k_chunk_mib * MiB);
    size_t idx  = pinned_chunk_pool::NO_CHUNK;
    if (!pool->configure_zones(MiB, MiB, MiB, k_chunk_mib * MiB) || !pool->zones_configured() ||
        !pool->grow_zone(host_zone_id::SCRATCH, 1, &idx) || idx == pinned_chunk_pool::NO_CHUNK ||
        !pool->withdraw_zone_chunk(host_zone_id::SCRATCH, idx)) {
        c.void_case("the child's own pool could not grow and withdraw a SCRATCH chunk");
        exit_now(c.status());
    }
    // Kept alive past the pool: never freed.
    auto * carrier = new pinned_chunk_pool::released_chunk;
    if (pool->release_zone_chunk(host_zone_id::SCRATCH, idx, *carrier) != PINNED_CHUNK_RELEASE_DONE ||
        !carrier->owns_backing()) {
        c.void_case("the empty withdrawn chunk was not released into a carrier");
        exit_now(c.status());
    }
    delete pool;
    c.fail("the pool's destruction returned with a released_chunk outstanding");
    exit_now(1);
    return 1;
}

int run_child_case(const child_args & a) {
    checks     c{ a.case_name.c_str() };
    const bool strict_wanted = a.case_name == "strict";
    if (strict_in_env() != strict_wanted) {
        c.void_case(strict_wanted ? "GGML_SYCL_STRICT_LEASES is unset in the strict child" :
                                    "GGML_SYCL_STRICT_LEASES is set in a non-strict child");
        return c.status();
    }
    // RLIMIT_CORE first: a core-dumping abort holds pinned memory past its exit message.
    if (!child_begin(a)) {
        return 2;
    }
    if (!entry_environment_ok(c)) {
        return c.status();
    }
    if (!strict_wanted && ggml_sycl_strict_enabled()) {
        c.fail("ggml_sycl_strict_enabled() is true at the plain child's entry");
        return c.status();
    }
    device_fixture f;
    if (!device_fixture_open(c, f)) {
        return c.status();
    }
    if (a.case_name == "strict") {
        return child_strict(c, f);
    }
    if (a.case_name == "plain") {
        const int rc = child_plain(c, f);
        ggml_backend_free(f.backend);
        return rc;
    }
    if (a.case_name == "carrier") {
        return child_carrier(c, f);
    }
    fprintf(stderr, "unknown child '%s'\n", a.case_name.c_str());
    return 2;
}

// --- the driver ---

// The driver's environ without the two variables the children set, then `extra`.
struct child_env {
    std::vector<std::string> strings;
    std::vector<char *>      envp;
};

void child_env_build(child_env & e, const std::vector<std::string> & extra) {
    for (char ** v = environ; *v != nullptr; ++v) {
        if (std::strncmp(*v, "GGML_SYCL_STRICT_LEASES=", 24) != 0 && std::strncmp(*v, "GGML_NO_BACKTRACE=", 18) != 0) {
            e.strings.emplace_back(*v);
        }
    }
    for (const std::string & s : extra) {
        e.strings.push_back(s);
    }
    for (std::string & s : e.strings) {
        e.envp.push_back(s.data());
    }
    e.envp.push_back(nullptr);
}

enum class verdict { PASS, FAIL, VOID };

std::string status_of(const child_run & r) {
    if (WIFSIGNALED(r.status)) {
        return "died by signal " + std::to_string(WTERMSIG(r.status));
    }
    return "exited " + std::to_string(WEXITSTATUS(r.status));
}

// A child that must abort with `line` on its stderr.
verdict score_abort(const char * who, const child_run & r, const char * line) {
    if (WIFEXITED(r.status) && WEXITSTATUS(r.status) == SYCL_TEST_SKIP) {
        fprintf(stderr, "VOID [%s] (exit 77)\n", who);
        return verdict::VOID;
    }
    if (WIFSIGNALED(r.status) && WTERMSIG(r.status) == SIGABRT) {
        if (r.err.find(line) == std::string::npos) {
            fprintf(stderr, "FAIL [%s] %s child aborted without the `%s` line\n", who, who, line);
            return verdict::FAIL;
        }
        fprintf(stderr, "PASS [%s]\n", who);
        return verdict::PASS;
    }
    fprintf(stderr, "FAIL [%s] %s child %s, expected SIGABRT\n", who, who, status_of(r).c_str());
    return verdict::FAIL;
}

verdict score_clean(const char * who, const child_run & r) {
    if (WIFEXITED(r.status) && WEXITSTATUS(r.status) == SYCL_TEST_SKIP) {
        fprintf(stderr, "VOID [%s] (exit 77)\n", who);
        return verdict::VOID;
    }
    if (WIFEXITED(r.status) && WEXITSTATUS(r.status) == 0) {
        fprintf(stderr, "PASS [%s]\n", who);
        return verdict::PASS;
    }
    fprintf(stderr, "FAIL [%s] %s child %s, expected exit 0\n", who, who, status_of(r).c_str());
    return verdict::FAIL;
}

int drive(const char * exe, const std::vector<const char *> & children, const std::string & mutants) {
    // Every envp is built here, before any fork.
    child_env strict_env;
    child_env plain_env;
    child_env_build(strict_env, { "GGML_SYCL_STRICT_LEASES=1", "GGML_NO_BACKTRACE=1" });
    child_env_build(plain_env, { "GGML_NO_BACKTRACE=1" });
    const std::chrono::milliseconds bound(90000);

    bool any_fail = false;
    bool any_void = false;
    for (const char * name : children) {
        const std::string        child = name;
        std::vector<std::string> args  = { exe, "--case", child };
        if (!mutants.empty()) {
            args.push_back("--mutant");
            args.push_back(mutants);
        }
        std::vector<char *> argv;
        for (std::string & s : args) {
            argv.push_back(s.data());
        }
        argv.push_back(nullptr);
        char * const *  envp = child == "strict" ? strict_env.envp.data() : plain_env.envp.data();
        const child_run r    = run_child(argv.data(), envp, bound);
        fputs(r.err.c_str(), stderr);
        // The design's names for the three children.
        const char * who = child == "carrier" ? "outstanding-carrier" : name;
        verdict      v   = verdict::FAIL;
        if (!r.ran) {
            fprintf(stderr, "FAIL [%s] the %s child did not finish within %lld ms\n", who, who,
                    (long long) bound.count());
        } else if (child == "strict") {
            v = score_abort(who, r, k_pool_refused_line);
        } else if (child == "plain") {
            v = score_clean(who, r);
        } else {
            v = score_abort(who, r, k_carrier_line);
        }
        any_fail = any_fail || v == verdict::FAIL;
        any_void = any_void || v == verdict::VOID;
    }
    if (any_fail) {
        return 1;
    }
    return any_void ? SYCL_TEST_SKIP : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const child_args a = child_args_parse(argc, argv);
    if (!a.case_name.empty()) {
        return run_child_case(a);
    }
    std::vector<const char *> selected;
    std::string               unknown;
    if (!cases_select({ "strict", "plain", "carrier" }, a.cases, &selected, &unknown)) {
        fprintf(stderr, "unknown child '%s'\n", unknown.c_str());
        return 2;
    }
    // The driver touches no device: each child runs in its own process.
    return drive(argv[0], selected, a.mutants);
}
