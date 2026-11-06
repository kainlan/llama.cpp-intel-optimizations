//
// resolve_or_host on a mapped weight aborts naming the tensor (docs/design/sycl-mmap-host-tier.md §5.3, §7 C4
// test-sycl-file-mapped-resolve-abort).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// A death test: its one case, "abort", runs in its own child. The child maps a fixture file through the FILE_MAPPED
// front door on device 0's cache, points a named tensor's data at the mapping, as a SYCL_Mapped weight's is, and calls
// ggml_sycl_resolve_or_host_tensor_ptr. A mapped page is no pointer a device may be handed, so the call must abort.
// The driver scores the child on rc 134 (SIGABRT) plus the tensor's name in its output. A child that returns from the
// call prints what it got and exits 1.

#include "../common.hpp"
#include "file-mapped-test-common.hpp"

#include <csignal>

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

constexpr const char * k_tensor_name = "blk.0.file_mapped_resolve_abort.weight";

int run_abort(checks & c) {
    if (!file_mapped_test::entry_ok(c)) {
        return c.status();
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    file_mapped_test::fixture_file f(4);
    if (!f.ok) {
        c.void_case("the fixture file could not be written");
        return c.status();
    }
    mem_handle mapping = file_mapped_test::map_whole(c, f, 0);
    if (!mapping.valid()) {
        return c.status();
    }
    const resolved_ptr view = mapping.resolve();
    if (!view || view.on_device || !alloc_registry::instance().is_file_mapped(view.ptr)) {
        c.void_case("the mapping did not resolve to a host pointer the registry names FILE_MAPPED");
        return c.status();
    }

    ggml_tensor t{};
    t.type  = GGML_TYPE_F32;
    t.ne[0] = static_cast<int64_t>(f.size / sizeof(float));
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0]                     = sizeof(float);
    t.nb[1] = t.nb[2] = t.nb[3] = f.size;
    t.data                      = view.ptr;
    snprintf(t.name, sizeof(t.name), "%s", k_tensor_name);

    fprintf(stderr, "[FILE-MAPPED-RESOLVE] resolving %s over the mapping\n", k_tensor_name);
    void * got = ggml_sycl_resolve_or_host_tensor_ptr(&t, 0);
    c.fail(std::string("resolve_or_host on mapped weight ") + k_tensor_name + " returned " +
           (got == view.ptr ? "the mapped pointer" :
            got == nullptr  ? "nullptr" :
                              "another pointer") +
           " instead of aborting");
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "abort") {
        return run_abort(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

// The child's SIGABRT is the pass, and only with the tensor's name in its output; a VOID stays a VOID.
struct abort_runner : scratch_return_test::case_runner {
    scratch_return_test::case_verdict score(const char *                                name,
                                            const host_ledger_test_harness::child_run & r) const override {
        if (WIFSIGNALED(r.status) && WTERMSIG(r.status) == SIGABRT) {
            if (r.err.find(k_tensor_name) == std::string::npos) {
                fprintf(stderr, "FAIL [%s] rc 134, but the abort does not name the tensor %s\n", name, k_tensor_name);
                return scratch_return_test::case_verdict::FAIL;
            }
            fprintf(stderr, "PASS [%s] rc 134 naming %s\n", name, k_tensor_name);
            return scratch_return_test::case_verdict::PASS;
        }
        if (!WIFSIGNALED(r.status) && WEXITSTATUS(r.status) == 0) {
            fprintf(stderr, "FAIL [%s] exit 0: the case returned without aborting\n", name);
            return scratch_return_test::case_verdict::FAIL;
        }
        return case_runner::score(name, r);
    }
};

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "abort" };
    const scratch_return_test::child_args  a   = scratch_return_test::child_args_parse(argc, argv);
    if (a.case_name.empty() && !file_mapped_test::control_gate()) {
        return SYCL_TEST_SKIP;
    }
    const abort_runner runner;
    return scratch_return_test::cases_main(argc, argv, all, run_case, std::chrono::minutes(2), runner);
}
