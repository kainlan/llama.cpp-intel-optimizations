//
// Test: mem_handle device identity and wrong-device resolution
// (llama.cpp-32dg8.15.11 — P2-FIX)
//
// Verifies:
//   (1) from_direct() no longer silently implies device 0: device_ matches the
//       device_id passed, or is HOST_DEVICE (-1) for host pointers.
//   (2) resolve(device_id) on same-device DIRECT handle returns the pointer.
//   (3) resolve(device_id) on wrong-device DIRECT handle returns null (explicit fail).
//   (4) HOST_DEVICE handles resolve successfully from any device (host-agnostic).
//   (5) [Multi-GPU only] Same checks for CHUNK_LEASE handles using from_chunk_ptr.
//   (6) [Multi-GPU only] WEIGHT handle wrong-device resolve returns null.
//
// Single-GPU systems: cases (5) and (6) are skipped with a clear message.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

#include "../mem-handle.hpp"
#include "../unified-cache.hpp"

// =============================================================================
// Test harness
// =============================================================================

static int g_tests_run    = 0;
static int g_tests_passed = 0;
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

#define TEST_SKIP(reason)                             \
    do {                                              \
        g_tests_skipped++;                            \
        g_tests_run--;                                \
        fprintf(stderr, "SKIPPED: %s\n", (reason));   \
        return true;                                  \
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

// =============================================================================
// Test 1: from_direct device_id is stored correctly
// =============================================================================
static bool test_from_direct_stores_device_id() {
    TEST_BEGIN("from_direct_stores_device_id");

    int  marker  = 0;
    void * ptr   = &marker;

    // Device handle: device_ should be 0
    ggml_sycl::mem_handle h_dev = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 0);
    TEST_ASSERT(h_dev.device() == 0, "from_direct device handle must store device 0");

    // Host handle: device_ should be HOST_DEVICE (-1)
    ggml_sycl::mem_handle h_host = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, false,
                                                                       ggml_sycl::mem_handle::HOST_DEVICE);
    TEST_ASSERT(h_host.device() == ggml_sycl::mem_handle::HOST_DEVICE,
                "from_direct host handle must store HOST_DEVICE");

    // Default device_id argument is HOST_DEVICE
    ggml_sycl::mem_handle h_default = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, false);
    TEST_ASSERT(h_default.device() == ggml_sycl::mem_handle::HOST_DEVICE,
                "from_direct default device must be HOST_DEVICE, not 0");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Same-device DIRECT resolve passes
// =============================================================================
static bool test_same_device_direct_resolve_passes() {
    TEST_BEGIN("same_device_direct_resolve_passes");

    int  marker = 42;
    void * ptr  = &marker;

    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 0);
    TEST_ASSERT(h.device() == 0, "handle device must be 0");

    ggml_sycl::resolved_ptr r = h.resolve(0);
    TEST_ASSERT(r.ptr == ptr, "same-device resolve must return the direct ptr");
    TEST_ASSERT(r.on_device, "on_device flag must be preserved");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Wrong-device DIRECT resolve returns null (explicit fail policy)
// =============================================================================
static bool test_wrong_device_direct_resolve_fails(int n_gpu_devices) {
    TEST_BEGIN("wrong_device_direct_resolve_fails");

    if (n_gpu_devices < 2) {
        TEST_SKIP("fewer than 2 GPU devices available");
    }

    int  marker = 99;
    void * ptr  = &marker;

    // Handle for device 0
    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 0);
    TEST_ASSERT(h.device() == 0, "handle device must be 0");

    // Resolve from device 1 — must fail (return null)
    ggml_sycl::resolved_ptr r = h.resolve(1);
    TEST_ASSERT(r.ptr == nullptr, "wrong-device DIRECT resolve must return null ptr");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: HOST_DEVICE handle resolves from any device
// =============================================================================
static bool test_host_device_handle_resolves_from_any_device(int n_gpu_devices) {
    TEST_BEGIN("host_device_handle_resolves_from_any_device");

    int  marker = 77;
    void * ptr  = &marker;

    // Host handle: device_ == HOST_DEVICE
    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, false,
                                                                  ggml_sycl::mem_handle::HOST_DEVICE);

    // Should resolve successfully from device 0
    ggml_sycl::resolved_ptr r0 = h.resolve(0);
    TEST_ASSERT(r0.ptr == ptr, "HOST_DEVICE handle must resolve from device 0");

    if (n_gpu_devices >= 2) {
        // Should also resolve from device 1
        ggml_sycl::resolved_ptr r1 = h.resolve(1);
        TEST_ASSERT(r1.ptr == ptr, "HOST_DEVICE handle must resolve from device 1");
    }

    // Should also resolve from device 99 (any arbitrary device — no ownership)
    ggml_sycl::resolved_ptr r99 = h.resolve(99);
    TEST_ASSERT(r99.ptr == ptr, "HOST_DEVICE handle must resolve from any device id");

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: wrong-device CHUNK_LEASE resolve returns null (multi-GPU)
//
// from_chunk_ptr on a device-0-allocated pointer should produce a handle with
// device_ == 0 (either CHUNK_LEASE or DIRECT). Resolving it from device 1 must fail.
// =============================================================================
static bool test_wrong_device_chunk_lease_fails(int n_gpu_devices) {
    TEST_BEGIN("wrong_device_chunk_lease_resolve_fails");

    if (n_gpu_devices < 2) {
        TEST_SKIP("fewer than 2 GPU devices available");
    }

    // We need a device-0 SYCL queue and a pointer that is in device-0's arena.
    // Use a minimal unified_cache to get a device pointer.
    std::vector<sycl::device> devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if ((int) devs.size() < 2) {
        TEST_SKIP("fewer than 2 SYCL GPU devices enumerated");
    }

    sycl::queue q0(devs[0], sycl::property::queue::in_order{});

    constexpr size_t bytes  = 4096;
    constexpr size_t budget = 16 * 1024 * 1024;
    ggml_sycl::unified_cache cache(q0, budget);

    void * dev_ptr = sycl::malloc_device(bytes, q0);
    if (!dev_ptr) {
        TEST_SKIP("sycl::malloc_device failed on device 0");
    }

    // from_chunk_ptr will produce a DIRECT or CHUNK_LEASE handle with device 0
    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_chunk_ptr(dev_ptr, 0, GGML_LAYOUT_AOS, true);
    TEST_ASSERT(h.device() == 0, "from_chunk_ptr handle must carry device 0");

    // Resolve from device 0 — must succeed
    ggml_sycl::resolved_ptr r0 = h.resolve(0);
    TEST_ASSERT(r0.ptr == dev_ptr, "same-device CHUNK/DIRECT resolve must return the ptr");

    // Resolve from device 1 — must fail
    ggml_sycl::resolved_ptr r1 = h.resolve(1);
    TEST_ASSERT(r1.ptr == nullptr, "wrong-device CHUNK/DIRECT resolve must return null");

    sycl::free(dev_ptr, q0);
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: wrong-device WEIGHT handle resolve returns null (multi-GPU)
// =============================================================================
static bool test_wrong_device_weight_handle_fails(int n_gpu_devices) {
    TEST_BEGIN("wrong_device_weight_handle_resolve_fails");

    if (n_gpu_devices < 2) {
        TEST_SKIP("fewer than 2 GPU devices available");
    }

    // Build a WEIGHT handle for device 0 with a bogus cache key (no entry in cache).
    // resolve(1) must return null without crashing — the wrong-device check fires
    // before any cache lookup.
    ggml_sycl_cache_id id = {};
    id.valid              = true;
    id.model_id           = 9999;
    id.aux_id             = 1;
    id.nbytes             = 1024;
    id.name_hash          = id.model_id ^ id.aux_id;
    id.type               = GGML_TYPE_F32;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = (i == 0) ? 256 : 1;
        id.tp_local_ne[i]  = id.ne[i];
        id.tp_offset_ne[i] = 0;
    }

    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_cache_id(id, 0);
    TEST_ASSERT(h.device() == 0, "WEIGHT handle must carry device 0");

    // Wrong-device check: resolve from device 1 via device-checking overload
    ggml_sycl::resolved_ptr r1 = h.resolve(1);
    TEST_ASSERT(r1.ptr == nullptr, "wrong-device WEIGHT resolve must return null");

    // Same-device: resolve from device 0 — cache miss returns null, but no crash
    ggml_sycl::resolved_ptr r0 = h.resolve(0);
    // r0.ptr may be null (no cache entry), that's fine — no assertion on ptr value
    (void) r0;

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=================================================\n");
    fprintf(stderr, "mem_handle device identity / wrong-device tests\n");
    fprintf(stderr, "(llama.cpp-32dg8.15.11 — P2-FIX)\n");
    fprintf(stderr, "=================================================\n");

    // Count available GPU devices
    int n_gpu_devices = 0;
    try {
        auto devs     = sycl::device::get_devices(sycl::info::device_type::gpu);
        n_gpu_devices = static_cast<int>(devs.size());
    } catch (...) {
        n_gpu_devices = 0;
    }
    fprintf(stderr, "GPU devices available: %d\n", n_gpu_devices);
    if (n_gpu_devices < 2) {
        fprintf(stderr, "NOTE: fewer than 2 GPU devices — multi-device tests will be skipped\n");
    }
    fprintf(stderr, "-------------------------------------------------\n");

    bool all_passed = true;
    all_passed &= test_from_direct_stores_device_id();
    all_passed &= test_same_device_direct_resolve_passes();
    all_passed &= test_wrong_device_direct_resolve_fails(n_gpu_devices);
    all_passed &= test_host_device_handle_resolves_from_any_device(n_gpu_devices);
    all_passed &= test_wrong_device_chunk_lease_fails(n_gpu_devices);
    all_passed &= test_wrong_device_weight_handle_fails(n_gpu_devices);

    fprintf(stderr, "-------------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n",
            g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
