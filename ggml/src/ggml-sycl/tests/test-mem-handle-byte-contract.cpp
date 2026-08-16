//
// Test: unregistered external pointers get authority ONLY from the operation's
// declared byte contract (llama.cpp-fxrg)
//
// Regression coverage for the P0 fixed in this same change: the buffer-interface
// get/set_tensor family (tiered_kv_buffer_*, split_buffer_*, tp_buffer_*,
// dev2dev_memcpy) minted mem_handles for CALLER-SUPPLIED host pointers via
// ggml_sycl_copy_handle_for_raw_ptr(ptr, layout, device) -- omitting the 4th
// parameter, operation_bytes (default 0). An unknown external pointer with no
// declared byte contract resolves to extent == 0, and mem_copy correctly
// refuses it (require_resolved_range). f23bf786b added that guard; 6faf4c7ec
// claimed to have migrated every call site to pass the byte contract and never
// touched this family -- this test is what would have caught that omission.
//
// This exercises the same underlying mechanism the fixed call sites use
// (ggml_sycl_memcpy_handle_for_raw_ptr, the function ggml_sycl_copy_handle_for_raw_ptr
// in ggml-sycl.cpp wraps) directly, since that wrapper is `static` to its TU and
// not linkable from a separate test binary.
//
// Verifies:
//   (1) An unregistered external host pointer minted with NO byte contract
//       resolves to extent == 0.
//   (2) The same pointer minted WITH a byte contract resolves to extent == bytes.
//   (3) mem_copy refuses (aborts via GGML_ABORT) when either endpoint has no
//       declared authority for the bytes being moved.
//   (4) mem_copy succeeds, and the bytes actually transfer correctly, when both
//       endpoints declare authority via the byte contract.
//
// Cases (1)-(2) are pure host-side logic and always run. Cases (3)-(4) need a
// live sycl::queue to submit through (queue_device_or_host() calls
// ggml_sycl_get_device_id_from_queue(), which needs a real queue reference) and
// are skipped on a device-less host -- but they do NO device buffer allocation:
// both endpoints in this test are HOST_DEVICE handles, so mem_copy_direct_submit
// takes its host-to-host std::memcpy fast path and never touches the GPU. This
// needs no GPU slot in the sense CLAUDE.md's coordination rules mean -- no model
// load, no device buffer object, nothing that grows TTM shmem.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../mem-ops.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <sycl/sycl.hpp>
#include <vector>

// =============================================================================
// Test harness (same shape as test-mem-handle-wrong-device.cpp)
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_SKIP(reason)                           \
    do {                                            \
        g_tests_skipped++;                          \
        g_tests_run--;                              \
        fprintf(stderr, "SKIPPED: %s\n", (reason)); \
        return true;                                \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

// Runs `operation` in a forked child and reports whether it terminated
// non-successfully (killed by a signal, or exited non-zero) -- i.e. whether a
// GGML_ABORT actually fired. Mirrors test-mem-ops.cpp's expect_fatal.
static bool expect_fatal(const std::function<void()> & operation) {
    const pid_t child = fork();
    if (child == 0) {
        operation();
        _exit(0);
    }
    if (child < 0) {
        return false;
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) {
        return false;
    }
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// =============================================================================
// Test 1: unregistered external pointer with NO byte contract resolves to
// extent == 0 -- this is the exact shape of the bug: the buffer-interface call
// sites minted handles this way, via a 3-arg call that left operation_bytes at
// its `= 0` default.
// =============================================================================
static bool test_no_contract_resolves_zero_extent() {
    TEST_BEGIN("no_contract_resolves_zero_extent");

    std::vector<uint8_t> buf(64, 0xAB);

    // fallback_unknown=false matches ggml_sycl_copy_handle_for_raw_ptr's own
    // fixed call into this function exactly -- see ggml-sycl.cpp.
    ggml_sycl::mem_handle handle =
        ggml_sycl_memcpy_handle_for_raw_ptr(buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                            /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                            /*trusted_extent=*/0);
    ggml_sycl::resolved_ptr resolved = handle.resolve();
    TEST_ASSERT(resolved.ptr != nullptr, "handle must still resolve to a non-null pointer");
    TEST_ASSERT(resolved.extent == 0, "unregistered pointer with no byte contract must have extent 0");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: the same pointer, minted WITH a byte contract, resolves to
// extent == bytes -- this is the fix.
// =============================================================================
static bool test_with_contract_resolves_declared_extent() {
    TEST_BEGIN("with_contract_resolves_declared_extent");

    std::vector<uint8_t> buf(64, 0xAB);
    constexpr size_t     bytes = 64;

    ggml_sycl::mem_handle handle =
        ggml_sycl_memcpy_handle_for_raw_ptr(buf.data(), /*fallback_device=*/-1, GGML_LAYOUT_AOS,
                                            /*fallback_on_device=*/false, /*fallback_unknown=*/false,
                                            /*trusted_extent=*/bytes);
    ggml_sycl::resolved_ptr resolved = handle.resolve();
    TEST_ASSERT(resolved.ptr == buf.data(), "handle must resolve to the original pointer");
    TEST_ASSERT(resolved.extent == bytes, "declared byte contract must become the resolved extent");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: mem_copy refuses when either endpoint has zero declared authority.
// Both endpoints here are HOST_DEVICE, so no device work is ever submitted --
// require_resolved_range fires before mem_copy_direct_submit would touch a
// queue. Needs a live sycl::queue reference regardless (queue_device_or_host
// dereferences it), so it is skipped device-less.
// =============================================================================
static bool test_mem_copy_refuses_zero_authority(int n_gpu_devices) {
    TEST_BEGIN("mem_copy_refuses_zero_authority");

    if (n_gpu_devices < 1) {
        TEST_SKIP("no GPU device available to construct a queue");
    }

    std::vector<uint8_t> src(64, 0xCD);
    std::vector<uint8_t> dst(64, 0x00);

    bool died = expect_fatal([&] {
        sycl::queue           q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        ggml_sycl::mem_handle dst_handle =
            ggml_sycl_memcpy_handle_for_raw_ptr(dst.data(), -1, GGML_LAYOUT_AOS, false, false, 0);
        ggml_sycl::mem_handle src_handle =
            ggml_sycl_memcpy_handle_for_raw_ptr(src.data(), -1, GGML_LAYOUT_AOS, false, false, 0);
        ggml_sycl::mem_copy(dst_handle, src_handle, 64, q);
    });
    TEST_ASSERT(died, "mem_copy must abort (GGML_ABORT via require_resolved_range) with zero authority");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: mem_copy is admitted, and the bytes actually move, when both
// endpoints declare the byte contract that matches what the copy moves.
// =============================================================================
static bool test_mem_copy_admits_declared_authority(int n_gpu_devices) {
    TEST_BEGIN("mem_copy_admits_declared_authority");

    if (n_gpu_devices < 1) {
        TEST_SKIP("no GPU device available to construct a queue");
    }

    constexpr size_t     bytes = 64;
    std::vector<uint8_t> src(bytes, 0xEF);
    std::vector<uint8_t> dst(bytes, 0x00);

    sycl::queue           q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    ggml_sycl::mem_handle dst_handle =
        ggml_sycl_memcpy_handle_for_raw_ptr(dst.data(), -1, GGML_LAYOUT_AOS, false, false, bytes);
    ggml_sycl::mem_handle src_handle =
        ggml_sycl_memcpy_handle_for_raw_ptr(src.data(), -1, GGML_LAYOUT_AOS, false, false, bytes);
    ggml_sycl::mem_copy(dst_handle, src_handle, bytes, q);

    TEST_ASSERT(std::memcmp(dst.data(), src.data(), bytes) == 0,
                "mem_copy with a correct byte contract must actually move the bytes");

    TEST_PASS();
    return true;
}

int main() {
    fprintf(stderr, "=================================================\n");
    fprintf(stderr, "mem_handle byte-contract authority tests\n");
    fprintf(stderr, "(llama.cpp-fxrg)\n");
    fprintf(stderr, "=================================================\n");

    const auto & sycl_info     = ggml_sycl_info();
    const int    n_gpu_devices = sycl_info.total_gpu_count;
    fprintf(stderr, "GPU devices available: %d physical, %d scheduler-visible\n", n_gpu_devices,
            sycl_info.device_count);
    fprintf(stderr, "-------------------------------------------------\n");

    bool all_passed = true;
    all_passed &= test_no_contract_resolves_zero_extent();
    all_passed &= test_with_contract_resolves_declared_extent();
    all_passed &= test_mem_copy_refuses_zero_authority(n_gpu_devices);
    all_passed &= test_mem_copy_admits_declared_authority(n_gpu_devices);

    fprintf(stderr, "-------------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n", g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
