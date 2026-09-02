#include "../../../../tests/test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"
#include "../unified-cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <sycl/sycl.hpp>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static bool expect_fatal(const char * label, const std::function<void()> & operation) {
    const pid_t child = fork();
    if (child == 0) {
        operation();
        _exit(0);
    }
    if (child < 0) {
        std::fprintf(stderr, "FAIL: %s fork failed\n", label);
        return false;
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || (WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
        std::fprintf(stderr, "FAIL: %s did not terminate fatally\n", label);
        return false;
    }
    return true;
}

static bool check_bytes(const uint8_t * data, size_t size, uint8_t expected, const char * label) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] != expected) {
            std::fprintf(stderr, "FAIL: %s byte %zu expected 0x%02x got 0x%02x\n", label, i, expected, data[i]);
            return false;
        }
    }
    return true;
}

int main() {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        // Exit 77, not 0. A device test that cannot reach a device has verified NOTHING,
        // and returning 0 made that indistinguishable from a pass -- run this binary
        // directly in a shell where setvars.sh was never sourced and it printed one line
        // and exited green in 0.4 s. 77 is ctest's SKIP_RETURN_CODE, so ctest reports
        // *skipped* rather than *passed*, and a bare shell run gets a non-zero status.
        // The point is to make the skip visible as a skip, not to forbid skipping: a
        // CPU-only runner still legitimately lands here. See llama.cpp-k208.
        //
        // ⚠️ 77 only reads as "skipped" once this test's registration carries
        // SKIP_RETURN_CODE 77; ggml/src/ggml-sycl/CMakeLists.txt:1274 does not yet.
        std::fprintf(stderr,
                     "SKIP: no SYCL GPU devices available -- NO DEVICE WORK WAS PERFORMED.\n"
                     "      On a machine that has a GPU this means the oneAPI runtime was not on the\n"
                     "      library path: source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    sycl::queue      q(gpus.front(), sycl::property::queue::in_order{});
    constexpr size_t size = 4096;

    uint8_t * host_a = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * host_b = sycl::malloc_host<uint8_t>(size, q);
    uint8_t * dev_a  = sycl::malloc_device<uint8_t>(size, q);
    uint8_t * dev_b  = sycl::malloc_device<uint8_t>(size, q);
    if (!host_a || !host_b || !dev_a || !dev_b) {
        std::fprintf(stderr, "FAIL: allocation failed\n");
        sycl::free(host_a, q);
        sycl::free(host_b, q);
        sycl::free(dev_a, q);
        sycl::free(dev_b, q);
        return 1;
    }

    int  failed   = 0;
    auto host_a_h = ggml_sycl::mem_handle::from_direct(
        host_a, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size);
    auto host_b_h = ggml_sycl::mem_handle::from_direct(
        host_b, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size);
    auto dev_a_h = ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size);
    auto dev_b_h = ggml_sycl::mem_handle::from_direct(dev_b, GGML_LAYOUT_AOS, true, 0, size);

    std::memset(host_a, 0x11, size);
    std::memset(host_b, 0x00, size);
    ggml_sycl::mem_copy(host_b_h, host_a_h, size, q);
    failed += !check_bytes(host_b, size, 0x11, "H2H copy");

    std::memset(host_a, 0x22, size);
    auto h2d = ggml_sycl::mem_copy_async(dev_a_h, host_a_h, size, q);
    auto d2h = ggml_sycl::mem_copy_async(host_b_h, dev_a_h, size, q, { h2d });
    d2h.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x22, "H2D/D2H copy");

    auto fill = ggml_sycl::mem_fill_async(dev_a_h, 0x33, size, q);
    auto d2d  = ggml_sycl::mem_copy_async(dev_b_h, dev_a_h, size, q, { fill });
    auto back = ggml_sycl::mem_copy_async(host_b_h, dev_b_h, size, q, { d2d });
    back.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x33, "fill+D2D+D2H chain");

    // Temporary handles may die immediately after submission.  mem_ops must
    // keep any handle-owned lease alive until the returned event completes.
    auto temp_fill = ggml_sycl::mem_fill_async(
        ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size), 0x44, size, q);
    auto temp_copy = ggml_sycl::mem_copy_async(
        ggml_sycl::mem_handle::from_direct(
            host_b, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE, size),
        ggml_sycl::mem_handle::from_direct(dev_a, GGML_LAYOUT_AOS, true, 0, size), size, q, { temp_fill });
    temp_copy.wait_and_throw();
    failed += !check_bytes(host_b, size, 0x44, "temporary handle event lifetime");

    // Exact boundary succeeds; one-byte offset and SIZE_MAX offset arithmetic
    // are rejected before any queue submission.
    ggml_sycl::mem_copy(host_b_h, size - 1, host_a_h, size - 1, 1, q);
    failed += !expect_fatal("source extent overflow", [&] {
        ggml_sycl::mem_copy(host_b_h, 0, host_a_h, size - 1, 2, q);
    });
    failed += !expect_fatal("destination offset overflow", [&] {
        ggml_sycl::mem_fill(host_b_h, SIZE_MAX, 0, 1, q);
    });

    ggml_sycl::alloc_request req{};
    req.queue                               = &q;
    req.size                                = size;
    req.intent.role                         = ggml_sycl::alloc_role::STAGING;
    req.intent.category                     = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::WEIGHT;

    ggml_sycl::alloc_handle unified_alloc{};
    if (!ggml_sycl::unified_alloc(req, &unified_alloc) || !unified_alloc.ptr) {
        std::fprintf(stderr, "FAIL: unified_alloc device buffer failed\n");
        failed++;
    } else {
        ggml_sycl::mem_handle unified_h    = ggml_sycl::detail::from_legacy_owned_alloc(std::move(unified_alloc));
        auto                  unified_fill = ggml_sycl::mem_fill_async(unified_h, 0x55, size, q);
        auto                  unified_back = ggml_sycl::mem_copy_async(host_b_h, unified_h, size, q, { unified_fill });
        unified_back.wait_and_throw();
        failed += !check_bytes(host_b, size, 0x55, "unified_alloc-owned fill+D2H");
    }

    // llama.cpp-4do9 (TKV-9): the census over tiered_kv_buffer_set_tensor and
    // the tp-buffer get/set functions found every ggml_sycl_copy_handle_for_raw_ptr
    // call site already declaring its byte contract (fixed by c1f4504c8; see
    // that commit's message for the family of sites this guards). This locks
    // in the underlying primitive's trusted-extent resolution directly: an
    // unregistered external pointer with NO declared byte contract must
    // resolve to extent 0 (the exact fxrg failure shape -- llama.cpp-fxrg),
    // and the same pointer WITH a byte contract must resolve to that declared
    // extent. It does NOT exercise the call sites above, and dropping the
    // `, size` argument at any of them would still leave this test green --
    // call-site discipline is source-review (census) territory, not something
    // this test can catch structurally.
    {
        std::vector<uint8_t> unregistered_buf(128, 0xAB);

        ggml_sycl::mem_handle no_contract =
            ggml_sycl_memcpy_handle_for_raw_ptr(unregistered_buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                                /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                                /*trusted_extent=*/0);
        ggml_sycl::resolved_ptr no_contract_resolved = no_contract.resolve();
        if (no_contract_resolved.extent != 0) {
            std::fprintf(stderr, "FAIL: unregistered pointer with no byte contract must resolve to extent 0, got %zu\n",
                         no_contract_resolved.extent);
            failed++;
        }

        constexpr size_t      contract_bytes = 128;
        ggml_sycl::mem_handle with_contract =
            ggml_sycl_memcpy_handle_for_raw_ptr(unregistered_buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                                /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                                /*trusted_extent=*/contract_bytes);
        ggml_sycl::resolved_ptr with_contract_resolved = with_contract.resolve();
        if (with_contract_resolved.ptr != unregistered_buf.data() || with_contract_resolved.extent != contract_bytes) {
            std::fprintf(stderr,
                         "FAIL: unregistered pointer with a %zu-byte contract must resolve to that extent "
                         "(got ptr=%p extent=%zu)\n",
                         contract_bytes, with_contract_resolved.ptr, with_contract_resolved.extent);
            failed++;
        }
    }

    sycl::free(host_a, q);
    sycl::free(host_b, q);
    sycl::free(dev_a, q);
    sycl::free(dev_b, q);

    if (failed) {
        std::fprintf(stderr, "FAIL: %d mem_ops checks failed\n", failed);
        return 1;
    }
    std::printf("PASS: mem_ops H2H/H2D/D2H/D2D/dependency checks passed\n");
    return 0;
}
