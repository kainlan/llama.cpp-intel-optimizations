//
// Which /proc/meminfo counter pinned host USM lands in (docs/design/sycl-mmap-host-tier.md §7 C4,
// test-sycl-pinned-meminfo-control; r2 m-13). The lead runs it before the other file-mapped REDs and before C4's
// host-growth criterion, which reads the counter this test names.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// One case. It allocates 4 GiB through unified_allocate with must_host_pinned (a standalone host-USM base, so the
// whole figure is one driver allocation and no pool chunk size rounds it), touches every page, samples AnonPages,
// Shmem, Mlocked, Unevictable and Cached before and after, prints each delta, and frees it. Pre-registered: at least
// one counter moves by at least 3.5 GiB, and that counter is the one pinned host USM lands in (run2's Shmem stayed at
// 1.8 G while 140 GB was pinned, so it is not Shmem). No counter moving exits 77: the host-growth criterion then rests
// on the Committed line and MemAvailable alone. The outcome is written beside the binary for the other five tests.

#include "file-mapped-test-common.hpp"

#include <cinttypes>

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

constexpr size_t k_bytes = 4ull << 30;     // 4 GiB
constexpr size_t k_moved = 3584ull << 20;  // 3.5 GiB

const char * const k_counters[] = { "AnonPages", "Shmem", "Mlocked", "Unevictable", "Cached" };
constexpr int      k_ncounters  = 5;

// Each counter's value in bytes; false when one is missing from /proc/meminfo.
bool meminfo_sample(int64_t (&out)[k_ncounters]) {
    std::ifstream in("/proc/meminfo");
    std::string   line;
    bool          seen[k_ncounters] = {};
    while (std::getline(in, line)) {
        for (int i = 0; i < k_ncounters; ++i) {
            const std::string key = std::string(k_counters[i]) + ":";
            if (line.compare(0, key.size(), key) == 0) {
                out[i]  = std::strtoll(line.c_str() + key.size(), nullptr, 10) * 1024;
                seen[i] = true;
            }
        }
    }
    for (bool s : seen) {
        if (!s) {
            return false;
        }
    }
    return true;
}

int run_meminfo(checks & c) {
    if (sycl_test_gpu_devices().empty()) {
        sycl_test_print_skip("the pinned meminfo control");
        c.void_case("no SYCL GPU device");
        return c.status();
    }
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (backend == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return c.status();
    }
    int64_t before[k_ncounters] = {};
    int64_t after[k_ncounters]  = {};
    if (!meminfo_sample(before)) {
        c.void_case("/proc/meminfo lacks one of AnonPages, Shmem, Mlocked, Unevictable, Cached");
        return c.status();
    }

    alloc_request req{};
    req.queue                                    = &cache->get_queue();
    req.device                                   = 0;
    req.size                                     = k_bytes;
    req.intent.role                              = alloc_role::STAGING;
    req.intent.category                          = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned      = true;
    req.intent.constraints.require_host_usm_base = true;
    {
        mem_handle         pinned = unified_allocate(req);
        const resolved_ptr view   = pinned.valid() ? pinned.resolve() : resolved_ptr{};
        if (!view || view.on_device) {
            c.void_case("unified_allocate refused a 4 GiB must_host_pinned request");
            return c.status();
        }
        const size_t    page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        volatile char * p    = static_cast<volatile char *>(view.ptr);
        for (size_t off = 0; off < k_bytes; off += page) {
            p[off] = static_cast<char>(off / page);
        }
        if (!meminfo_sample(after)) {
            c.void_case("/proc/meminfo lost a counter between the samples");
            return c.status();
        }
    }

    int     moved    = -1;
    int64_t moved_by = 0;
    for (int i = 0; i < k_ncounters; ++i) {
        const int64_t delta = after[i] - before[i];
        fprintf(stderr, "[MEMINFO-CONTROL] %s %+.1f MiB\n", k_counters[i], delta / (1024.0 * 1024.0));
        if (delta >= static_cast<int64_t>(k_moved) && delta > moved_by) {
            moved    = i;
            moved_by = delta;
        }
    }
    if (moved < 0) {
        c.void_case(
            "no counter moved by 3.5 GiB over 4 GiB of touched pinned host USM: the host-growth criterion "
            "rests on the Committed line and MemAvailable alone");
        return c.status();
    }
    fprintf(stderr, "[MEMINFO-CONTROL] pinned host USM lands in %s (%+.1f MiB for 4096.0 MiB touched)\n",
            k_counters[moved], moved_by / (1024.0 * 1024.0));
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "meminfo") {
        return run_meminfo(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

// The driver's verdict is the control's outcome, written for the other five before it exits.
struct control_runner : scratch_return_test::case_runner {
    scratch_return_test::case_verdict score(const char *                                name,
                                            const host_ledger_test_harness::child_run & r) const override {
        const scratch_return_test::case_verdict v = case_runner::score(name, r);
        file_mapped_test::control_outcome_write(v == scratch_return_test::case_verdict::PASS ? "PASS" :
                                                v == scratch_return_test::case_verdict::VOID ? "VOID" :
                                                                                               "FAIL");
        return v;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "meminfo" };
    const control_runner                   runner;
    return scratch_return_test::cases_main(argc, argv, all, run_case, std::chrono::minutes(5), runner);
}
