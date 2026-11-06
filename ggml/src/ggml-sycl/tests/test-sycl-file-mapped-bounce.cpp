//
// mem_copy of a mapped source takes the bounce (docs/design/sycl-mmap-host-tier.md §5.3, §7 C4
// test-sycl-file-mapped-bounce).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Case "bounce": it maps a fixture file through the FILE_MAPPED front door on device 0's cache and copies the whole
// mapping into a device RUNTIME buffer with mem_copy. A mapped page is not USM (host_ptr_is_usm_accessible is false
// for it), so the copy must take the staged host->device branch: the submitting thread's staged-copy census rises by
// exactly one, and the device buffer reads back the file's bytes.

#include "../mem-ops.hpp"
#include "file-mapped-test-common.hpp"

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

int run_bounce(checks & c) {
    if (!file_mapped_test::entry_ok(c)) {
        return c.status();
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return c.status();
    }
    unified_cache * cache = get_unified_cache_for_device(0);
    if (cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return c.status();
    }
    file_mapped_test::fixture_file f(16);
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

    sycl::queue & q = cache->get_queue();
    alloc_request req{};
    req.queue                                     = &q;
    req.device                                    = 0;
    req.size                                      = f.size;
    req.intent.role                               = alloc_role::STAGING;
    req.intent.category                           = runtime_category::STAGING;
    req.intent.constraints.must_device            = true;
    req.intent.constraints.prefer_vram_zone       = vram_zone_id::RUNTIME;
    req.intent.constraints.forbid_vram_zone_spill = true;
    mem_handle dst                                = unified_allocate(req);
    if (!dst.valid()) {
        c.void_case("the RUNTIME zone refused the device buffer");
        return c.status();
    }

    const uint64_t before = mem_copy_staged_wait_census_this_thread().staged_h2d;
    mem_copy(dst, mapping, f.size, q);
    const uint64_t after = mem_copy_staged_wait_census_this_thread().staged_h2d;
    c.expect(after == before + 1, "mem_copy of a FILE_MAPPED source took " + std::to_string(after - before) +
                                      " staged host->device branch(es), expected 1: the bounce was not taken");

    std::vector<uint8_t> out(f.size, 0);
    mem_copy(mem_handle::from_direct(out.data(), GGML_LAYOUT_AOS, false, mem_handle::HOST_DEVICE, f.size), dst, f.size,
             q);
    for (size_t i = 0; i < f.size; ++i) {
        if (out[i] != file_mapped_test::pattern(i)) {
            c.fail("the device buffer reads byte " + std::to_string(i) + " back wrong after the bounce");
            break;
        }
    }
    fprintf(stderr, "[FILE-MAPPED-BOUNCE] staged h2d %llu -> %llu\n", static_cast<unsigned long long>(before),
            static_cast<unsigned long long>(after));
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "bounce") {
        return run_bounce(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "bounce" };
    return file_mapped_test::gated_main(argc, argv, "test-sycl-file-mapped-bounce", all, run_case,
                                        std::chrono::minutes(2));
}
