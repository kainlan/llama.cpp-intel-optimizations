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
//   (5) Real CHUNK_LEASE handle via unified_cache_host_zone_alloc(SCRATCH) into global
//       cache's pinned pool; kind()==CHUNK_LEASE tripwire (all systems); wrong-device
//       resolve(1)==null (multi-GPU only).
//   (6) [Multi-GPU only] WEIGHT handle wrong-device resolve returns null.
//
// No-GPU systems: device-backed cases skip before touching unified cache. Single-GPU
// systems: case (5) runs the CHUNK_LEASE tripwire but skips the wrong-device resolve(1)
// check via inline NOTE.  Case (6) is skipped entirely.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../common.hpp"
#include "../mem-handle.hpp"
#include "../unified-cache.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"

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

    // Same-device resolve must pass before testing wrong-device failure
    ggml_sycl::resolved_ptr r0 = h.resolve(0);
    TEST_ASSERT(r0.ptr == ptr, "same-device DIRECT resolve must return the ptr");

    // Resolve from device 1 — must fail (return null)
    ggml_sycl::resolved_ptr r1 = h.resolve(1);
    TEST_ASSERT(r1.ptr == nullptr, "wrong-device DIRECT resolve must return null ptr");

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
// Uses unified_cache_host_zone_alloc(SCRATCH) to get a pointer that lives
// inside the global cache's pinned_chunk_pool.  from_chunk_ptr then hits
// host_acquire_chunk_lease and reliably produces CHUNK_LEASE (not DIRECT)
// regardless of VRAM pressure.  The CHUNK_LEASE handle stores device_=0
// (the device argument to from_chunk_ptr), so resolve(1) fails with the
// wrong-device policy.
//
// Tripwire: kind() == CHUNK_LEASE is asserted before the resolve tests so
// any future DIRECT fallthrough is caught immediately rather than silently
// re-testing the wrong-device policy on the same path as test 3.
// =============================================================================
static bool test_chunk_lease_tripwire_and_wrong_device_resolve(int n_gpu_devices) {
    TEST_BEGIN("chunk_lease_tripwire_and_wrong_device_resolve");

    if (n_gpu_devices == 0) {
        TEST_SKIP("no GPU devices available");
    }

    // Allocate a small buffer through the global cache's pinned host pool.
    // unified_cache_host_zone_alloc → host_zone_alloc → host_arena_->allocate_segmented
    // (or zone_alloc_segmented if zones are configured).  Either way the returned
    // ptr is inside a pinned_chunk_pool chunk whose range is known to
    // host_acquire_chunk_lease.  This is a deterministic path — no VRAM pressure
    // dependency, no H2D copy, no staging buffer needed.
    constexpr size_t bytes = 256;
    void * host_ptr = ggml_sycl::unified_cache_host_zone_alloc(ggml_sycl::host_zone_id::SCRATCH, bytes, 64);
    if (!host_ptr) {
        TEST_SKIP("unified_cache_host_zone_alloc returned null (cache not available)");
    }

    // from_chunk_ptr with device=0: host_acquire_chunk_lease finds host_ptr inside
    // the global cache's pinned_chunk_pool → CHUNK_LEASE handle with device_=0.
    ggml_sycl::mem_handle h = ggml_sycl::mem_handle::from_chunk_ptr(host_ptr, 0,
                                                                      GGML_LAYOUT_AOS, false);
    TEST_ASSERT(h.device() == 0, "from_chunk_ptr handle must carry device 0");

    // Tripwire: fires regardless of GPU count — verifies the ptr was found in the
    // pinned pool and from_chunk_ptr produced a real CHUNK_LEASE, not a DIRECT fallthrough.
    TEST_ASSERT(h.kind() == ggml_sycl::mem_handle_kind::CHUNK_LEASE,
                "from_chunk_ptr must produce CHUNK_LEASE for ptr in pinned pool");

    // Same-device resolve must return the pointer.
    ggml_sycl::resolved_ptr r0 = h.resolve(0);
    TEST_ASSERT(r0.ptr == host_ptr, "same-device CHUNK_LEASE resolve must return the ptr");

    if (n_gpu_devices < 2) {
        // Wrong-device resolve can only be tested when a second device exists
        // (resolve(1) needs a valid device_id to check against).
        fprintf(stderr, "  [NOTE] wrong-device resolve(1) skipped — fewer than 2 GPUs\n");
    } else {
        // Wrong-device resolve must return null (explicit-fail policy).
        // handle's device_=0 != caller device_id=1, so the check fires and returns null.
        ggml_sycl::resolved_ptr r1 = h.resolve(1);
        TEST_ASSERT(r1.ptr == nullptr, "wrong-device CHUNK_LEASE resolve must return null");
    }

    // Note: host_ptr lives in the SCRATCH zone which is reset-only (no per-alloc
    // free). The handle dtor releases the chunk lease; the pool reclaims on zone reset.
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
// Test 7: mem_handle equality/hash are usable for dispatch route tables
// =============================================================================
static bool test_mem_handle_hash_identity() {
    TEST_BEGIN("mem_handle_hash_identity");

    int marker = 123;
    void * ptr = &marker;

    ggml_sycl::mem_handle a = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 0);
    ggml_sycl::mem_handle b = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_SOA, true, 0);
    TEST_ASSERT(a == b, "direct handles resolving to the same allocation must compare equal");
    TEST_ASSERT(a.hash() == b.hash(), "direct handles resolving to the same allocation must hash the same");

    std::unordered_set<ggml_sycl::mem_handle, ggml_sycl::mem_handle_hash> handles;
    handles.insert(a);
    handles.insert(b);
    TEST_ASSERT(handles.size() == 1, "unordered_set must collapse handles for the same allocation");

    std::unordered_set<ggml_sycl::mem_handle> default_hash_handles;
    default_hash_handles.insert(a);
    default_hash_handles.insert(b);
    TEST_ASSERT(default_hash_handles.size() == 1, "std::hash<mem_handle> must key same-allocation aliases");

    ggml_sycl::mem_handle c = a;
    TEST_ASSERT(c == a, "copied direct handle must compare equal to source");
    TEST_ASSERT(c.hash() == a.hash(), "copied direct handle must hash equal to source");

    std::unordered_map<ggml_sycl::mem_handle, int> route_map;
    route_map.emplace(a, 17);
    TEST_ASSERT(route_map.find(c) != route_map.end(), "copied same-allocation handle must find existing map entry");

    ggml_sycl::mem_handle host_a =
        ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, false, ggml_sycl::mem_handle::HOST_DEVICE);
    ggml_sycl::mem_handle host_b =
        ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_COALESCED, false, ggml_sycl::mem_handle::HOST_DEVICE);
    TEST_ASSERT(host_a == host_b, "host handles resolving to same allocation must compare equal");
    TEST_ASSERT(host_a.hash() == host_b.hash(), "host handles resolving to same allocation must hash the same");

    ggml_sycl_cache_id id = {};
    id.valid              = true;
    id.model_id           = 8888;
    id.aux_id             = 7;
    id.nbytes             = 4096;
    id.name_hash          = id.model_id ^ id.aux_id;
    id.type               = GGML_TYPE_Q4_0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = (i == 0) ? 1024 : 1;
        id.tp_local_ne[i]  = id.ne[i];
        id.tp_offset_ne[i] = 0;
    }

    ggml_sycl::mem_handle w0 = ggml_sycl::mem_handle::from_cache_id(id, 0);
    ggml_sycl::mem_handle w1 = ggml_sycl::mem_handle::from_cache_id(id, 0);
    TEST_ASSERT(w0 == w1, "unresolved weight handles with the same cache identity must compare equal");
    TEST_ASSERT(w0.hash() == w1.hash(), "unresolved weight handles with the same cache identity must hash the same");

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

    const auto & sycl_info = ggml_sycl_info();
    const int    n_gpu_devices = sycl_info.total_gpu_count;
    fprintf(stderr, "GPU devices available: %d physical, %d scheduler-visible\n",
            n_gpu_devices, sycl_info.device_count);
    if (n_gpu_devices < 2) {
        fprintf(stderr, "NOTE: fewer than 2 GPU devices — multi-device tests will be skipped\n");
    }
    fprintf(stderr, "-------------------------------------------------\n");

    // Initialize SYCL backend for device 0 so g_device_caches[0] is populated.
    // Test 5 needs unified_cache_host_zone_alloc which queries the global registry.
    // The backend object can be freed immediately; the g_device_caches entry persists.
    if (sycl_info.device_count > 0) {
        ggml_backend_t backend = ggml_backend_sycl_init(0);
        if (backend) {
            ggml_backend_free(backend);
        }
    }
    fprintf(stderr, "-------------------------------------------------\n");

    bool all_passed = true;
    all_passed &= test_from_direct_stores_device_id();
    all_passed &= test_same_device_direct_resolve_passes();
    all_passed &= test_wrong_device_direct_resolve_fails(n_gpu_devices);
    all_passed &= test_host_device_handle_resolves_from_any_device(n_gpu_devices);
    all_passed &= test_chunk_lease_tripwire_and_wrong_device_resolve(n_gpu_devices);
    all_passed &= test_wrong_device_weight_handle_fails(n_gpu_devices);
    all_passed &= test_mem_handle_hash_identity();

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
