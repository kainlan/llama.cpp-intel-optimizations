//
// resolve_or_host on a mapped weight aborts naming the tensor (docs/design/sycl-mmap-host-tier.md §5.3, §7 C4
// test-sycl-file-mapped-resolve-abort).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// A death test with one case per branch of ggml_sycl_resolve_or_host_tensor_ptr, each in its own child. Both map a
// fixture file through the FILE_MAPPED front door on device 0's cache, point a named tensor's data at the mapping, as a
// SYCL_Mapped weight's is, and call the resolver. A mapped page is no pointer a device may be handed, so the call must
// abort. The two branches print the same text, so each case first shows which branch it reaches:
//   abort      the tensor resolves to a pointer (ggml_sycl_resolve_tensor_ptr returns the mapping), so the resolved
//              branch's check is the one that must abort;
//   abort-raw  a view whose root is owned by another device resolves to null on device 0, so the host-data branch's
//              check is the one that must abort. The other device's allocation is a registry entry over a host array,
//              as tests/test-sycl-kv-view-resolution.cpp builds one; nothing is allocated on a second device.
// The driver scores each child on rc 134 (SIGABRT) plus the abort's printed text, which names that case's tensor. A
// child that returns from the call prints what it got and exits 1.

#include "../common.hpp"
#include "file-mapped-test-common.hpp"

#include <csignal>

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

constexpr const char * k_tensor_name     = "blk.0.file_mapped_resolve_abort.weight";
constexpr const char * k_raw_tensor_name = "blk.0.file_mapped_resolve_abort_raw.weight";
// The text each case's abort prints. GGML_ABORT is consumed by the preprocessor and never reaches a log, so the
// printed message is what is scored.
constexpr const char * k_abort_text =
    "[FILE-MAPPED] resolve_or_host on mapped weight blk.0.file_mapped_resolve_abort.weight";
constexpr const char * k_raw_abort_text =
    "[FILE-MAPPED] resolve_or_host on mapped weight blk.0.file_mapped_resolve_abort_raw.weight";

const char * abort_text_for(const std::string & case_name) {
    return case_name == "abort-raw" ? k_raw_abort_text : k_abort_text;
}

// The mapping both cases point a tensor at. Invalid, with the case VOID, when any step refuses.
struct mapped_fixture {
    file_mapped_test::fixture_file file{ 4 };
    mem_handle                     mapping;
    resolved_ptr                   view;
};

bool mapped_fixture_open(checks & c, mapped_fixture & m) {
    if (!file_mapped_test::entry_ok(c)) {
        return false;
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return false;
    }
    if (!m.file.ok) {
        c.void_case("the fixture file could not be written");
        return false;
    }
    m.mapping = file_mapped_test::map_whole(c, m.file, 0);
    if (!m.mapping.valid()) {
        return false;
    }
    m.view = m.mapping.resolve();
    if (!m.view || m.view.on_device || !alloc_registry::instance().is_file_mapped(m.view.ptr)) {
        c.void_case("the mapping did not resolve to a host pointer the registry names FILE_MAPPED");
        return false;
    }
    return true;
}

void tensor_over(ggml_tensor & t, const mapped_fixture & m, const char * name) {
    t.type  = GGML_TYPE_F32;
    t.ne[0] = static_cast<int64_t>(m.file.size / sizeof(float));
    t.ne[1] = t.ne[2] = t.ne[3] = 1;
    t.nb[0]                     = sizeof(float);
    t.nb[1] = t.nb[2] = t.nb[3] = m.file.size;
    t.data                      = m.view.ptr;
    snprintf(t.name, sizeof(t.name), "%s", name);
}

void fail_returned(checks & c, const char * name, const void * got, const void * mapped) {
    c.fail(std::string("resolve_or_host on mapped weight ") + name + " returned " +
           (got == mapped  ? "the mapped pointer" :
            got == nullptr ? "nullptr" :
                             "another pointer") +
           " instead of aborting");
}

int run_abort(checks & c) {
    mapped_fixture m;
    if (!mapped_fixture_open(c, m)) {
        return c.status();
    }
    ggml_tensor t{};
    tensor_over(t, m, k_tensor_name);
    // The resolved branch: the plain resolver hands back a pointer, so its check, not the host-data one, decides.
    if (ggml_sycl_resolve_tensor_ptr(&t, 0) == nullptr) {
        c.void_case("the tensor resolved to null, so the case would reach the host-data branch, not the resolved one");
        return c.status();
    }

    fprintf(stderr, "[FILE-MAPPED-RESOLVE] resolving %s over the mapping\n", k_tensor_name);
    void * got = ggml_sycl_resolve_or_host_tensor_ptr(&t, 0);
    fail_returned(c, k_tensor_name, got, m.view.ptr);
    return c.status();
}

int run_abort_raw(checks & c) {
    mapped_fixture m;
    if (!mapped_fixture_open(c, m)) {
        return c.status();
    }
    // The root's storage on device 1, as the registry records it: a view of the root then has no pointer on device 0.
    alignas(64) static unsigned char other_device_storage[256];
    alloc_registry::instance().register_alloc(other_device_storage, sizeof(other_device_storage), 1,
                                              alloc_type::DEVICE);
    ggml_tensor_extra_gpu root_extra{};
    root_extra.data_device[1] = other_device_storage;

    ggml_tensor root{};
    tensor_over(root, m, "blk.0.file_mapped_resolve_abort_raw.root");
    root.extra = &root_extra;
    ggml_tensor t{};
    tensor_over(t, m, k_raw_tensor_name);
    t.view_src = &root;

    // The host-data branch: the plain resolver returns null, so only the check on the tensor's own data can abort.
    if (ggml_sycl_resolve_tensor_ptr(&t, 0) != nullptr) {
        c.void_case(
            "the view resolved to a pointer, so the case would reach the resolved branch, not the host-data one");
        alloc_registry::instance().unregister_alloc(other_device_storage);
        return c.status();
    }

    fprintf(stderr, "[FILE-MAPPED-RESOLVE] resolving %s over the mapping\n", k_raw_tensor_name);
    void * got = ggml_sycl_resolve_or_host_tensor_ptr(&t, 0);
    fail_returned(c, k_raw_tensor_name, got, m.view.ptr);
    alloc_registry::instance().unregister_alloc(other_device_storage);
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
    if (a.case_name == "abort-raw") {
        return run_abort_raw(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

// The child's SIGABRT is the pass, and only with its case's abort text in its output; a VOID stays a VOID.
struct abort_runner : scratch_return_test::case_runner {
    scratch_return_test::case_verdict score(const char *                                name,
                                            const host_ledger_test_harness::child_run & r) const override {
        const char * text = abort_text_for(name);
        if (WIFSIGNALED(r.status) && WTERMSIG(r.status) == SIGABRT) {
            if (r.err.find(text) == std::string::npos) {
                fprintf(stderr, "FAIL [%s] rc 134, but the abort does not print \"%s\"\n", name, text);
                return scratch_return_test::case_verdict::FAIL;
            }
            fprintf(stderr, "PASS [%s] rc 134 printing \"%s\"\n", name, text);
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
    static const std::vector<const char *> all = { "abort", "abort-raw" };
    const abort_runner                     runner;
    return file_mapped_test::gated_main(argc, argv, all, run_case, std::chrono::minutes(2), runner);
}
