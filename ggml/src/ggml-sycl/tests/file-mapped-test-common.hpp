//
// What the file-mapped tier's device REDs share (docs/design/sycl-mmap-host-tier.md §7 C4:
// test-sycl-seg-refusal-cleanup, test-sycl-file-mapped-stream-refusal, test-sycl-file-mapped-bounce,
// test-sycl-file-mapped-expert-owner, test-sycl-file-mapped-resolve-abort, test-sycl-pinned-meminfo-control): the
// control's outcome file and the gate the other five read it through, and a mapped fixture file.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// test-sycl-pinned-meminfo-control runs first (the lead's order, §7 C4). It writes its outcome, PASS, VOID or FAIL,
// to a file beside its binary, and each of the other five reads that file in its driver, before any child forks or
// any device is touched: a missing file or a FAIL prints one VOID line naming the control's outcome and the test exits
// 77, so a test run before the control, or after it failed, never reads as a pass or a fail of its own. The file also
// carries the boot it was written in, and an outcome from another boot gates nothing: the file outlives the run, so
// without it a targeted run after a reboot (a new driver, another card state) would read an old PASS.

#pragma once

#include "../unified-cache.hpp"
#include "scratch-return-test-common.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace file_mapped_test {

using namespace ggml_sycl;
using scratch_return_test::checks;

constexpr const char * k_control_name    = "test-sycl-pinned-meminfo-control";
constexpr const char * k_control_outcome = "sycl-pinned-meminfo-control.outcome";

// The control's outcome file: beside the running binary, so every test of one build reads the same file.
inline std::string control_outcome_path() {
    char      exe[PATH_MAX];
    const int n = static_cast<int>(readlink("/proc/self/exe", exe, sizeof(exe) - 1));
    if (n <= 0) {
        return k_control_outcome;
    }
    exe[n]                 = '\0';
    const std::string path = exe;
    const size_t      cut  = path.rfind('/');
    return (cut == std::string::npos ? std::string(".") : path.substr(0, cut)) + "/" + k_control_outcome;
}

// This boot's id, from the kernel, or "unknown" when it cannot be read.
inline std::string boot_id_current() {
    std::ifstream in("/proc/sys/kernel/random/boot_id");
    std::string   id;
    if (!(in >> id)) {
        return "unknown";
    }
    return id;
}

// The control writes its outcome, "PASS", "VOID" or "FAIL", and the boot it ran in.
inline void control_outcome_write(const char * outcome) {
    FILE * f = fopen(control_outcome_path().c_str(), "w");
    if (f != nullptr) {
        fprintf(f, "%s\n%s\n", outcome, boot_id_current().c_str());
        fclose(f);
    }
}

// The control's outcome and the boot it was written in, as written; "missing" for an absent or empty file, and an
// empty boot for a file written before the boot was recorded.
struct control_outcome {
    std::string outcome;
    std::string boot;
};

inline control_outcome control_outcome_read() {
    std::ifstream   in(control_outcome_path());
    control_outcome r;
    if (!(in >> r.outcome)) {
        r.outcome = "missing";
        return r;
    }
    in >> r.boot;
    return r;
}

// The gate each of the five runs first. true when the control ran in this boot and did not FAIL (its VOID, no counter
// moved, only moves the lead's host-growth criterion to the Committed line and leaves these tests' mechanisms
// reachable).
inline bool control_gate() {
    const control_outcome r = control_outcome_read();
    if (r.outcome != "PASS" && r.outcome != "VOID") {
        fprintf(stderr, "VOID: %s outcome %s; run it first\n", k_control_name, r.outcome.c_str());
        return false;
    }
    const std::string boot = boot_id_current();
    if (r.boot != boot) {
        fprintf(stderr, "VOID: %s outcome %s is from boot %s, not this boot %s; run it first\n", k_control_name,
                r.outcome.c_str(), r.boot.empty() ? "(unrecorded)" : r.boot.c_str(), boot.c_str());
        return false;
    }
    return true;
}

// The registration pins one device; a run without it, or without a GPU, measures nothing.
inline bool entry_ok(checks & c) {
    const char * selector = getenv("ONEAPI_DEVICE_SELECTOR");
    if (selector == nullptr || selector[0] == '\0') {
        c.void_case("ONEAPI_DEVICE_SELECTOR is unset: the registration pins one device");
        return false;
    }
    if (sycl_test_gpu_devices().empty()) {
        sycl_test_print_skip("the file-mapped tier");
        c.void_case("no SYCL GPU device");
        return false;
    }
    return true;
}

// The main of each of the five: a child runs its case; the driver first reads the control's outcome and, when the
// gate refuses, exits 77 with the gate's one line and forks nothing.
inline int gated_main(int                               argc,
                      char **                           argv,
                      const std::vector<const char *> & all,
                      int (*run_case)(const scratch_return_test::child_args &),
                      std::chrono::milliseconds bound) {
    const scratch_return_test::child_args a = scratch_return_test::child_args_parse(argc, argv);
    if (a.case_name.empty() && !control_gate()) {
        return SYCL_TEST_SKIP;
    }
    return scratch_return_test::cases_main(argc, argv, all, run_case, bound);
}

// A file of `pages` pages whose byte i is pattern(i), open read-only. It is unlinked at destruction; a mapping of it
// keeps the inode alive until its last reference unmaps.
inline unsigned char pattern(uint64_t i) {
    return static_cast<unsigned char>(i * 131 + 7);
}

struct fixture_file {
    std::string path;
    int         fd   = -1;
    uint64_t    size = 0;
    bool        ok   = false;

    explicit fixture_file(uint64_t pages) {
        char      tmpl[] = "/tmp/file-mapped-red-XXXXXX";
        const int tmp    = mkstemp(tmpl);
        if (tmp < 0) {
            return;
        }
        path                = tmpl;
        const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        size                = pages * page;
        std::vector<unsigned char> bytes(size);
        for (uint64_t i = 0; i < size; ++i) {
            bytes[i] = pattern(i);
        }
        const bool written = write(tmp, bytes.data(), size) == static_cast<ssize_t>(size);
        close(tmp);
        fd = open(path.c_str(), O_RDONLY);
        ok = written && fd >= 0;
    }

    ~fixture_file() {
        if (fd >= 0) {
            close(fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

// Is any VMA of this process mapping the file `path`?
inline bool maps_file(const std::string & path) {
    std::ifstream in("/proc/self/maps");
    std::string   line;
    while (std::getline(in, line)) {
        if (line.find(path) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// The whole file mapped through the FILE_MAPPED front door on `device`'s cache, as one reservation and one segment,
// wrapped in a mem_handle: the reference a SYCL_Mapped buffer holds. Invalid, with the case VOID, on a refusal.
inline mem_handle map_whole(checks & c, const fixture_file & f, int device) {
    file_mapping_request req;
    req.device      = device;
    req.fd          = f.fd;
    req.file_size   = f.size;
    req.reservation = { 0, f.size };
    req.segments    = {
        { 0, f.size }
    };
    allocation_result mapped = unified_allocate_file_mapping(req);
    if (!mapped) {
        c.void_case("unified_allocate_file_mapping refused a whole-file request over " + f.path);
        return {};
    }
    mem_handle h = mem_handle::from_owned_alloc(std::move(mapped.owner));
    if (!h.valid()) {
        c.void_case("the FILE_MAPPED owner did not wrap into a mem_handle");
    }
    return h;
}

}  // namespace file_mapped_test
