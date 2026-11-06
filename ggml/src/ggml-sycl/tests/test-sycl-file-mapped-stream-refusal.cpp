//
// stream_dma over a FILE_MAPPED view refuses, with and without a copy_fn, and the counter counts
// (docs/design/sycl-mmap-host-tier.md §5.3 "The stream_dma refusal", §7 C4 test-sycl-file-mapped-stream-refusal).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Two cases, each in its own child: with-copy-fn passes a custom copy_fn, without-copy-fn passes none. Each maps a
// fixture file through the FILE_MAPPED front door on device 0's cache, streams the whole view, and requires the typed
// refusal (refused_file_mapped, not ok, nothing submitted), no call of either callback, and the process-wide refusal
// counter one higher. The callbacks touch no memory, so a stream the refusal fails to stop reads no mapped page from a
// device. Mutant refusal-after-copy-fn (a runtime seam in stream_dma) tests the copy_fn before the refusal; it fails
// the with-copy-fn case.

#include "file-mapped-test-common.hpp"

using namespace ggml_sycl;
using scratch_return_test::checks;

namespace {

int g_slice_calls = 0;
int g_copy_calls  = 0;

sycl::event count_slice(sycl::queue &, void *, size_t, size_t, const void *, const std::vector<sycl::event> &) {
    ++g_slice_calls;
    return sycl::event{};
}

sycl::event count_copy(sycl::queue &,
                       void *,
                       size_t,
                       size_t,
                       const void *,
                       size_t,
                       const void *,
                       const std::vector<sycl::event> &) {
    ++g_copy_calls;
    return sycl::event{};
}

int run_stream(checks & c, bool with_copy_fn) {
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

    cache_ptr_view src{};
    src.ptr      = view.ptr;
    src.size     = f.size;
    src.location = cache_location::HOST_MMAP;

    const uint64_t                   before = unified_cache_stream_dma_file_mapped_refusals();
    unified_cache::dma_stream_result r =
        cache->stream_dma(src, f.size, 0, 0, count_slice, nullptr, {}, with_copy_fn ? count_copy : nullptr);
    const uint64_t after = unified_cache_stream_dma_file_mapped_refusals();

    const std::string arm = with_copy_fn ? "with a copy_fn" : "without a copy_fn";
    if (!r.refused_file_mapped) {
        c.fail("stream_dma over a FILE_MAPPED view " + arm +
               " was not refused: refused_file_mapped=0, copy_fn called " + std::to_string(g_copy_calls) +
               " time(s), slice_fn called " + std::to_string(g_slice_calls) + " time(s), refusals +" +
               std::to_string(after - before));
        return c.status();
    }
    c.expect(!r.ok && !r.submitted, "the refusal " + arm + " reported ok=" + std::to_string(r.ok) +
                                        " submitted=" + std::to_string(r.submitted));
    c.expect(g_copy_calls == 0 && g_slice_calls == 0, "the refusal " + arm + " still called copy_fn " +
                                                          std::to_string(g_copy_calls) + " and slice_fn " +
                                                          std::to_string(g_slice_calls) + " time(s)");
    c.expect(after == before + 1,
             "the refusal " + arm + " moved the counter by " + std::to_string(after - before) + ", expected 1");
    fprintf(stderr, "[FILE-MAPPED-STREAM] %s refused, counter %llu -> %llu\n", arm.c_str(),
            static_cast<unsigned long long>(before), static_cast<unsigned long long>(after));
    return c.status();
}

int run_case(const scratch_return_test::child_args & a) {
    if (!scratch_return_test::child_begin(a)) {
        return 2;
    }
    checks c{ a.case_name.c_str() };
    if (a.case_name == "with-copy-fn") {
        return run_stream(c, true);
    }
    if (a.case_name == "without-copy-fn") {
        return run_stream(c, false);
    }
    fprintf(stderr, "unknown case '%s'\n", a.case_name.c_str());
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    static const std::vector<const char *> all = { "with-copy-fn", "without-copy-fn" };
    return file_mapped_test::gated_main(argc, argv, all, run_case, std::chrono::minutes(2));
}
