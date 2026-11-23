//
// mem_copy of a mapped source takes the bounce, and the outside-materialization counter counts a device-bound copy of
// mapped bytes at each named chokepoint outside a materialization and none inside one
// (docs/design/sycl-mmap-host-tier.md §5.3 "Gates", §7 C4 test-sycl-file-mapped-bounce).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Each case maps a fixture file through the FILE_MAPPED front door on device 0's cache and copies the mapping into a
// device RUNTIME buffer.
// - "bounce": mem_copy. A mapped page is not USM (host_ptr_is_usm_accessible is false for it), so the copy takes the
//   staged host->device branch: the submitting thread's staged-copy census rises by exactly one, and the device buffer
//   reads back the file's bytes.
// - "inside-materialization": the same mem_copy with a file_mapped_materialization_scope open on this thread moves
//   the counter by 0.
// - "outside": one copy through each chokepoint, with no scope open and no load transaction, moves the counter by
//   exactly 1: mem_copy, mem_copy_async, mem_copy_ptr_async and unified_cache::copy_to_device_async. stream_dma refuses
//   a mapped source before it copies, so its chokepoint moves the counter by 0 here. Then the `[FILE-MAPPED] counters:`
//   line prints, and its two fields read the two counters.

#include "../mem-ops.hpp"
#include "file-mapped-test-common.hpp"

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

struct bounce_fixture {
    unified_cache *                cache = nullptr;
    file_mapped_test::fixture_file file{ 16 };
    mem_handle                     mapping;
    void *                         mapped = nullptr;
    mem_handle                     dst;
    void *                         dst_ptr = nullptr;
};

// The mapping and a device buffer of its size. False, with the case VOID, when any of them is missing.
bool bounce_open(checks & c, bounce_fixture & f) {
    if (!file_mapped_test::entry_ok(c)) {
        return false;
    }
    if (ggml_backend_sycl_init(0) == nullptr) {
        c.void_case("ggml_backend_sycl_init(0) failed");
        return false;
    }
    f.cache = get_unified_cache_for_device(0);
    if (f.cache == nullptr) {
        c.void_case("device 0 has no unified cache");
        return false;
    }
    if (!f.file.ok) {
        c.void_case("the fixture file could not be written");
        return false;
    }
    f.mapping = file_mapped_test::map_whole(c, f.file, 0);
    if (!f.mapping.valid()) {
        return false;
    }
    const resolved_ptr view = f.mapping.resolve();
    if (!view || view.on_device || !alloc_registry::instance().is_file_mapped(view.ptr)) {
        c.void_case("the mapping did not resolve to a host pointer the registry names FILE_MAPPED");
        return false;
    }
    f.mapped = view.ptr;

    alloc_request req{};
    req.queue                                     = &f.cache->get_queue();
    req.device                                    = 0;
    req.size                                      = f.file.size;
    req.intent.role                               = alloc_role::STAGING;
    req.intent.category                           = runtime_category::STAGING;
    req.intent.constraints.must_device            = true;
    req.intent.constraints.prefer_vram_zone       = vram_zone_id::RUNTIME;
    req.intent.constraints.forbid_vram_zone_spill = true;
    f.dst                                         = unified_allocate(req);
    const resolved_ptr dst_view                   = f.dst.valid() ? f.dst.resolve() : resolved_ptr{};
    if (!dst_view || !dst_view.on_device) {
        c.void_case("the RUNTIME zone refused the device buffer");
        return false;
    }
    f.dst_ptr = dst_view.ptr;
    return true;
}

int run_bounce(checks & c) {
    bounce_fixture f;
    if (!bounce_open(c, f)) {
        return c.status();
    }
    sycl::queue &  q      = f.cache->get_queue();
    const uint64_t before = mem_copy_staged_wait_census_this_thread().staged_h2d;
    mem_copy(f.dst, f.mapping, f.file.size, q);
    const uint64_t after = mem_copy_staged_wait_census_this_thread().staged_h2d;
    c.expect(after == before + 1, "mem_copy of a FILE_MAPPED source took " + std::to_string(after - before) +
                                      " staged host->device branch(es), expected 1: the bounce was not taken");

    std::vector<uint8_t> out(f.file.size, 0);
    mem_copy(mem_handle::from_direct(out.data(), GGML_LAYOUT_AOS, false, mem_handle::HOST_DEVICE, f.file.size), f.dst,
             f.file.size, q);
    for (size_t i = 0; i < f.file.size; ++i) {
        if (out[i] != file_mapped_test::pattern(i)) {
            c.fail("the device buffer reads byte " + std::to_string(i) + " back wrong after the bounce");
            break;
        }
    }
    fprintf(stderr, "[FILE-MAPPED-BOUNCE] staged h2d %llu -> %llu\n", static_cast<unsigned long long>(before),
            static_cast<unsigned long long>(after));
    return c.status();
}

int run_inside(checks & c) {
    bounce_fixture f;
    if (!bounce_open(c, f)) {
        return c.status();
    }
    const uint64_t before = unified_cache_file_mapped_device_copies_outside_materialization();
    {
        file_mapped_materialization_scope materialization;
        mem_copy(f.dst, f.mapping, f.file.size, f.cache->get_queue());
    }
    const uint64_t after = unified_cache_file_mapped_device_copies_outside_materialization();
    c.expect(after == before, "mem_copy inside a materialization moved the outside-materialization counter by " +
                                  std::to_string(after - before) + ", expected 0");
    return c.status();
}

int run_outside(checks & c) {
    bounce_fixture f;
    if (!bounce_open(c, f)) {
        return c.status();
    }
    sycl::queue &  q     = f.cache->get_queue();
    const size_t   bytes = f.file.size;
    const uint64_t start = unified_cache_file_mapped_device_copies_outside_materialization();

    struct chokepoint {
        const char * name;
        uint64_t     expected;
    };

    auto copy_through = [&](const char * name) {
        if (std::strcmp(name, "mem_copy") == 0) {
            mem_copy(f.dst, f.mapping, bytes, q);
        } else if (std::strcmp(name, "mem_copy_async") == 0) {
            mem_copy_async(f.dst, f.mapping, bytes, q).wait_and_throw();
        } else if (std::strcmp(name, "mem_copy_ptr_async") == 0) {
            mem_copy_ptr_async(f.dst_ptr, f.mapped, bytes, q).wait_and_throw();
        } else if (std::strcmp(name, "copy_to_device_async") == 0) {
            f.cache->copy_to_device_async_for_testing(f.dst_ptr, f.mapped, bytes).wait_and_throw();
        } else {
            cache_ptr_view src{};
            src.ptr      = f.mapped;
            src.size     = bytes;
            src.location = cache_location::HOST_MMAP;
            const unified_cache::dma_stream_result r =
                f.cache->stream_dma(src, bytes, 0, 0,
                                    [](sycl::queue &, void *, size_t, size_t, const void *,
                                       const std::vector<sycl::event> &) { return sycl::event{}; },
                                    nullptr, {});
            c.expect(r.refused_file_mapped, "stream_dma did not refuse the FILE_MAPPED source");
        }
    };
    const chokepoint points[] = {
        { "mem_copy",             1 },
        { "mem_copy_async",       1 },
        { "mem_copy_ptr_async",   1 },
        { "copy_to_device_async", 1 },
        { "stream_dma",           0 },
    };
    for (const chokepoint & p : points) {
        const uint64_t before = unified_cache_file_mapped_device_copies_outside_materialization();
        copy_through(p.name);
        const uint64_t after = unified_cache_file_mapped_device_copies_outside_materialization();
        c.expect(after - before == p.expected, std::string(p.name) +
                                                   " of a FILE_MAPPED source outside a materialization moved the "
                                                   "counter by " +
                                                   std::to_string(after - before) + ", expected " +
                                                   std::to_string(p.expected));
    }
    const uint64_t copies   = unified_cache_file_mapped_device_copies_outside_materialization();
    const uint64_t refusals = unified_cache_stream_dma_file_mapped_refusals();

    unified_cache_file_mapped_counters_log();
    const std::vector<std::string> lines = host_ledger_test_harness::log_lines_containing("[FILE-MAPPED] counters:");
    if (lines.empty()) {
        c.fail("no [FILE-MAPPED] counters: line");
        return c.status();
    }
    const std::string & line = lines.back();
    c.expect(host_ledger_test_harness::printed(line, "stream_dma_refusals") == refusals,
             "the counters line's stream_dma_refusals does not read the refusal counter " + std::to_string(refusals));
    c.expect(host_ledger_test_harness::printed(line, "device_copies_outside_materialization") == copies,
             "the counters line's device_copies_outside_materialization does not read the counter " +
                 std::to_string(copies));
    fprintf(stderr, "[FILE-MAPPED-BOUNCE] outside-materialization counter %llu -> %llu\n",
            static_cast<unsigned long long>(start), static_cast<unsigned long long>(copies));
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
    if (a.case_name == "inside-materialization") {
        return run_inside(c);
    }
    if (a.case_name == "outside") {
        return run_outside(c);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "bounce", "inside-materialization", "outside" };
    return file_mapped_test::gated_main(argc, argv, all, run_case, std::chrono::minutes(2));
}
