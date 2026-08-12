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
#include <limits>
#include <thread>
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

    void * host_ptr_offset = static_cast<char *>(host_ptr) + 64;
    ggml_sycl::mem_handle h_offset =
        ggml_sycl::mem_handle::from_chunk_ptr(host_ptr_offset, 0, GGML_LAYOUT_AOS, false);
    TEST_ASSERT(h_offset.kind() == ggml_sycl::mem_handle_kind::CHUNK_LEASE,
                "from_chunk_ptr must produce CHUNK_LEASE for offset ptr in pinned pool");
    TEST_ASSERT(!h.stable_identity_equal(h_offset),
                "CHUNK_LEASE stable identity must distinguish different ptrs inside the same leased chunk");
    TEST_ASSERT(h.stable_identity_hash() != h_offset.stable_identity_hash(),
                "CHUNK_LEASE stable hash must distinguish different ptrs inside the same leased chunk");

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

static bool test_mem_handle_stable_identity() {
    TEST_BEGIN("mem_handle_stable_identity");

    int    marker = 123;
    void * ptr    = &marker;

    ggml_sycl::mem_handle device_a = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 0);
    ggml_sycl::mem_handle device_b = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_SOA, true, 0);
    TEST_ASSERT(device_a.stable_identity_equal(device_b),
                "same-device DIRECT handles over the same external pointer must share stable identity");
    TEST_ASSERT(device_a.stable_identity_hash() == device_b.stable_identity_hash(),
                "stable hash must match stable identity equality");

    ggml_sycl::mem_handle other_device = ggml_sycl::mem_handle::from_direct(ptr, GGML_LAYOUT_AOS, true, 1);
    TEST_ASSERT(!device_a.stable_identity_equal(other_device), "device owner participates in DIRECT stable identity");

    ggml_sycl::mem_handle arena_a = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 8192, 0, 7, 700, 12288);
    ggml_sycl::mem_handle arena_b = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 8192, 0, 7, 700, 12288);
    ggml_sycl::mem_handle arena_new_gen = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 8192, 0, 8, 700, 12288);
    ggml_sycl::mem_handle arena_collision = ggml_sycl::mem_handle::from_arena_zone(
        static_cast<int>(ggml_sycl::vram_zone_id::RUNTIME), 4096, 8192, 0, 7, 701, 12288);
    TEST_ASSERT(arena_a.stable_identity_equal(arena_b),
                "arena handles with same owner/zone/offset/size/generation must share stable identity");
    TEST_ASSERT(arena_a.stable_identity_hash() == arena_b.stable_identity_hash(),
                "arena stable hash must match stable identity equality");
    TEST_ASSERT(!arena_a.stable_identity_equal(arena_new_gen),
                "arena generation participates in retained-cache stable identity");
    TEST_ASSERT(!arena_a.stable_identity_equal(arena_collision),
                "distinct allocator-minted IDs never alias despite identical arena coordinates");

    ggml_sycl_cache_id id = {};
    id.valid              = true;
    id.model_id           = 4242;
    id.aux_id             = 9;
    id.nbytes             = 2048;
    id.name_hash          = id.model_id ^ id.aux_id;
    id.type               = GGML_TYPE_Q8_0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i]           = (i == 0) ? 256 : 1;
        id.tp_local_ne[i]  = id.ne[i];
        id.tp_offset_ne[i] = 0;
    }

    ggml_sycl::mem_handle weight_a = ggml_sycl::mem_handle::from_cache_id(id, 0);
    ggml_sycl::mem_handle weight_b = ggml_sycl::mem_handle::from_cache_id(id, 0);
    id.aux_id++;
    ggml_sycl::mem_handle weight_other = ggml_sycl::mem_handle::from_cache_id(id, 0);
    TEST_ASSERT(weight_a.stable_identity_equal(weight_b),
                "weight handles with same cache ID must share stable identity");
    TEST_ASSERT(weight_a.stable_identity_hash() == weight_b.stable_identity_hash(),
                "weight stable hash must match stable identity equality");
    TEST_ASSERT(!weight_a.stable_identity_equal(weight_other), "weight cache ID participates in stable identity");

    std::unordered_set<ggml_sycl::mem_handle,
                       ggml_sycl::mem_handle_stable_identity_hash,
                       ggml_sycl::mem_handle_stable_identity_equal>
        stable_set;
    stable_set.insert(weight_a);
    TEST_ASSERT(stable_set.find(weight_b) != stable_set.end(),
                "stable identity hasher/equality must support retained cache lookup by equivalent mem_handle");
    TEST_ASSERT(stable_set.find(weight_other) == stable_set.end(),
                "stable identity hasher/equality must reject different backing identities");

    TEST_PASS();
    return true;
}

// =============================================================================
// Main
// =============================================================================

static bool test_cache_entry_minted_retention_identity() {
    TEST_BEGIN("cache_entry_minted_retention_identity");

    ggml_sycl_cache_id id{};
    id.valid     = true;
    id.model_id  = 9001;
    id.name_hash = 9002;
    id.nbytes    = 64;
    id.type      = GGML_TYPE_Q8_0;
    ggml_sycl::unified_cache_key key{ ggml_sycl::cache_entry_type::DENSE_WEIGHT, id, -1, -1 };

    ggml_sycl::unified_cache_entry first{};
    ggml_sycl::unified_cache_entry second{};
    int                            first_storage  = 1;
    int                            second_storage = 2;
    first.device_ptr   = &first_storage;
    first.size         = sizeof(first_storage);
    first.layout       = GGML_LAYOUT_AOS;
    first.location     = ggml_sycl::cache_location::HOST_PINNED;
    first.host_resident = true;
    first.owner_device = ggml_sycl::mem_handle::HOST_DEVICE;
    second.device_ptr   = &second_storage;
    second.size         = sizeof(second_storage);
    second.layout       = GGML_LAYOUT_AOS;
    second.location     = ggml_sycl::cache_location::HOST_PINNED;
    second.host_resident = true;
    second.owner_device = ggml_sycl::mem_handle::HOST_DEVICE;
    first.in_use_count.fetch_add(1);
    second.in_use_count.fetch_add(1);
    auto first_handle = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        key, ggml_sycl::mem_handle::HOST_DEVICE, &first_storage, GGML_LAYOUT_AOS, false, &first, {}, false,
        sycl::event{});
    auto second_handle = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        key, ggml_sycl::mem_handle::HOST_DEVICE, &second_storage, GGML_LAYOUT_AOS, false, &second, {}, false,
        sycl::event{});
    TEST_ASSERT(first_handle.valid() && second_handle.valid(),
                "authoritative cache fixtures must mint valid weight capabilities");
    TEST_ASSERT(first.has_retention_identity() && second.has_retention_identity() &&
                    first.allocation_identity() != second.allocation_identity(),
                "fresh/recreated backing must mint a distinct allocation capability");

    // Forgery rejection: an entry capability cannot label another pointer.
    first.in_use_count.fetch_add(1);
    auto forged = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        key, ggml_sycl::mem_handle::HOST_DEVICE, &second_storage, GGML_LAYOUT_AOS, false, &first, {}, false,
        sycl::event{});
    TEST_ASSERT(!forged.valid(), "cache capability factory must reject a pointer not owned by the leased entry");

    // Forged device is ignored: the entry is authoritative.
    first.in_use_count.fetch_add(1);
    auto forged_device = ggml_sycl::mem_handle::from_weight_lease_snapshot(
        key, 77, &first_storage, GGML_LAYOUT_AOS, false, &first, {}, false, sycl::event{});
    TEST_ASSERT(forged_device.device() == first.owner_device,
                "caller device must not relabel a cache-owned capability");

    TEST_PASS();
    return true;
}

static ggml_sycl_cache_id make_transition_id(uint64_t tag, size_t size) {
    ggml_sycl_cache_id id{};
    id.valid     = true;
    id.model_id  = 9100 + tag;
    id.aux_id    = tag;
    id.name_hash = 0xabc000 + tag;
    id.nbytes    = size;
    id.type      = GGML_TYPE_F32;
    id.ne[0]     = static_cast<int64_t>(size / sizeof(float));
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        id.ne[i] = 1;
    }
    return id;
}

static void release_weight_lease(ggml_sycl::unified_cache::weight_ptr_lease_result & lease) {
    if (lease.entry) {
        lease.entry->in_use_count.fetch_sub(1);
        lease.entry = nullptr;
    }
}

static bool test_retention_transitions_and_exhaustion(sycl::queue & q) {
    TEST_BEGIN("retention_transitions_and_exhaustion");

    constexpr size_t small_size = 4096;
    constexpr size_t large_size = 8192;
    std::vector<unsigned char> first(small_size, 0x11);
    std::vector<unsigned char> second(small_size, 0x22);
    std::vector<unsigned char> large(large_size, 0x33);
    std::vector<unsigned char> large_changed(large_size, 0x44);
    ggml_sycl::unified_cache   cache(q, 64 * 1024);
    const auto                 id = make_transition_id(1, small_size);

    void * initial_ptr = cache.ensure_cached(
        id, first.data(), first.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, true);
    TEST_ASSERT(initial_ptr != nullptr, "initial cache publication must succeed");
    q.wait_and_throw();
    auto initial = cache.acquire_weight_lease(id);
    TEST_ASSERT(initial && initial.allocation_id != 0 && initial.replacement_generation != 0,
                "initial entry must publish a complete capability");
    const uint64_t initial_alloc = initial.allocation_id;
    const uint64_t initial_gen   = initial.replacement_generation;
    release_weight_lease(initial);

    // register_ready is completion, not adoption: a caller cannot change the
    // extent while retaining the same pointer/capability.
    cache.register_ready(id, initial_ptr, GGML_LAYOUT_AOS, small_size + 1,
                         ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, first.data());
    auto extent_check = cache.acquire_weight_lease(id);
    TEST_ASSERT(extent_check && extent_check.ptr == initial_ptr && extent_check.allocation_extent == small_size &&
                    extent_check.allocation_id == initial_alloc && extent_check.replacement_generation == initial_gen,
                "same pointer with a forged extent must leave the old capability unchanged");
    release_weight_lease(extent_check);

    // A live lease forbids both byte replacement and reallocation.
    auto live = cache.acquire_weight_lease(id);
    TEST_ASSERT(live, "live-lease setup must acquire the entry");
    TEST_ASSERT(cache.ensure_cached(id, second.data(), second.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1,
                                    -1, GGML_LAYOUT_AOS, true) == nullptr,
                "recopy must refuse a live lease");
    TEST_ASSERT(cache.ensure_cached(id, large.data(), large.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1,
                                    -1, GGML_LAYOUT_AOS, true) == nullptr,
                "realloc must refuse a live lease");
    TEST_ASSERT(live.ptr == initial_ptr && live.allocation_id == initial_alloc &&
                    live.replacement_generation == initial_gen,
                "live-lease refusal must preserve the old backing and generation");
    release_weight_lease(live);

    // Successful recopy preserves allocation identity and advances generation.
    TEST_ASSERT(cache.ensure_cached(id, second.data(), second.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1,
                                    -1, GGML_LAYOUT_AOS, true) == initial_ptr,
                "same-extent recopy must reuse the backing");
    q.wait_and_throw();
    auto recopied = cache.acquire_weight_lease(id);
    TEST_ASSERT(recopied && recopied.allocation_id == initial_alloc &&
                    recopied.replacement_generation != initial_gen,
                "recopy must advance only replacement generation");
    release_weight_lease(recopied);

    // Successful realloc atomically replaces pointer, extent and both identities.
    void * large_ptr = cache.ensure_cached(
        id, large.data(), large.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, true);
    TEST_ASSERT(large_ptr != nullptr, "extent-changing realloc must succeed");
    q.wait_and_throw();
    auto reallocated = cache.acquire_weight_lease(id);
    TEST_ASSERT(reallocated && reallocated.ptr == large_ptr && reallocated.allocation_extent == large_size &&
                    reallocated.allocation_id != initial_alloc,
                "realloc must publish a new complete backing capability");
    const uint64_t realloc_alloc = reallocated.allocation_id;
    release_weight_lease(reallocated);

    // Eviction followed by recreation must never resurrect the old identity.
    TEST_ASSERT(cache.evict(1) >= large_size, "entry must be evictable after its lease is released");
    TEST_ASSERT(cache.lookup(id, GGML_LAYOUT_AOS) == nullptr, "evicted entry must disappear from lookup");
    large_ptr = cache.ensure_cached(
        id, large.data(), large.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, true);
    TEST_ASSERT(large_ptr != nullptr, "evicted entry must be recreatable");
    q.wait_and_throw();
    auto recreated = cache.acquire_weight_lease(id);
    TEST_ASSERT(recreated && recreated.allocation_id != realloc_alloc,
                "recreated backing must mint a fresh allocation identity");
    release_weight_lease(recreated);

    // Prepare an allocate_slot/register_ready transition before permanently
    // exhausting the process-wide mint.
    const auto slot_id  = make_transition_id(2, 1024);
    void *     slot_ptr = cache.allocate_slot(slot_id, 1024, GGML_LAYOUT_AOS);
    TEST_ASSERT(slot_ptr != nullptr, "slot setup must allocate");
    cache.register_ready(slot_id, slot_ptr, GGML_LAYOUT_AOS, 1024,
                         ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, first.data());
    auto ready_before = cache.acquire_weight_lease(slot_id);
    TEST_ASSERT(ready_before, "register_ready setup must publish the slot");
    const uint64_t ready_gen = ready_before.replacement_generation;
    release_weight_lease(ready_before);

    constexpr size_t thread_count = 8;
    constexpr size_t per_thread   = 128;
    std::vector<std::vector<uint64_t>> ids(thread_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (size_t t = 0; t < thread_count; ++t) {
        workers.emplace_back([&, t] {
            ids[t].reserve(per_thread);
            for (size_t i = 0; i < per_thread; ++i) {
                ids[t].push_back(ggml_sycl::unified_cache_mint_retention_identity());
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    std::unordered_set<uint64_t> unique;
    for (const auto & batch : ids) {
        for (uint64_t minted : batch) {
            TEST_ASSERT(minted != 0, "concurrent mint must never produce zero before exhaustion");
            TEST_ASSERT(unique.insert(minted).second, "concurrent mint must never reuse an identity");
        }
    }

    const size_t cache_used_before = cache.used();
    const size_t host_used_before  = ggml_sycl::unified_cache_get_runtime_host_bytes();

    // Exact boundary: UINT64_MAX is a sentinel, while MAX-2 and MAX-1 may each
    // be issued exactly once. The monotonic seam cannot rewind afterwards.
    constexpr uint64_t max_id = std::numeric_limits<uint64_t>::max();
    ggml_sycl::unified_cache_advance_retention_identity_counter_for_test(max_id - 2);
    TEST_ASSERT(ggml_sycl::unified_cache_mint_retention_identity() == max_id - 2,
                "near-exhaustion must issue MAX-2 exactly once");
    TEST_ASSERT(ggml_sycl::unified_cache_mint_retention_identity() == max_id - 1,
                "near-exhaustion must issue MAX-1 exactly once");
    TEST_ASSERT(ggml_sycl::unified_cache_mint_retention_identity() == 0,
                "UINT64_MAX sentinel must never be issued");
    ggml_sycl::unified_cache_advance_retention_identity_counter_for_test(1);
    TEST_ASSERT(ggml_sycl::unified_cache_mint_retention_identity() == 0,
                "test seam must not rewind an exhausted minter");
    ggml_sycl::unified_cache_exhaust_retention_identities_for_test();
    TEST_ASSERT(ggml_sycl::unified_cache_mint_retention_identity() == 0 &&
                    ggml_sycl::unified_cache_mint_retention_identity() == 0,
                "exhausted mint must remain permanently exhausted and never wrap");

    const auto refused_slot_id = make_transition_id(3, 1024);
    TEST_ASSERT(cache.allocate_slot(refused_slot_id, 1024, GGML_LAYOUT_AOS) == nullptr,
                "allocate_slot must refuse zero identity before allocation");
    TEST_ASSERT(cache.used() == cache_used_before && cache.lookup(refused_slot_id, GGML_LAYOUT_AOS) == nullptr,
                "refused slot must roll back without budget or publication changes");

    cache.register_ready(slot_id, slot_ptr, GGML_LAYOUT_AOS, 1024,
                         ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, second.data());
    auto ready_after = cache.acquire_weight_lease(slot_id);
    TEST_ASSERT(ready_after && ready_after.ptr == slot_ptr && ready_after.replacement_generation == ready_gen &&
                    ready_after.entry->src_ptr == first.data(),
                "register_ready identity exhaustion must preserve old bytes metadata and generation");
    release_weight_lease(ready_after);

    auto old = cache.acquire_weight_lease(id);
    TEST_ASSERT(old, "exhaustion rollback setup must find recreated entry");
    const void *   old_ptr    = old.ptr;
    const uint64_t old_alloc  = old.allocation_id;
    const uint64_t old_gen    = old.replacement_generation;
    const size_t   old_extent = old.allocation_extent;
    release_weight_lease(old);
    TEST_ASSERT(cache.ensure_cached(id, large_changed.data(), large_changed.size(),
                                    ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, true) ==
                    nullptr,
                "recopy must fail closed when generation mint is exhausted");
    TEST_ASSERT(cache.ensure_cached(id, first.data(), first.size(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1,
                                    -1, GGML_LAYOUT_AOS, true) == nullptr,
                "realloc/host-fallback path must fail before allocation when identity is exhausted");
    auto preserved = cache.acquire_weight_lease(id);
    TEST_ASSERT(preserved && preserved.ptr == old_ptr && preserved.allocation_id == old_alloc &&
                    preserved.replacement_generation == old_gen && preserved.allocation_extent == old_extent,
                "failed recopy/realloc/fallback transitions must preserve old backing and generation");
    release_weight_lease(preserved);
    TEST_ASSERT(ggml_sycl::unified_cache_get_runtime_host_bytes() == host_used_before,
                "identity refusal must not allocate host fallback storage");

    ggml_sycl::alloc_request req{};
    req.queue                               = &q;
    req.size                                = 1024;
    req.intent.role                         = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                     = ggml_sycl::runtime_category::COMPUTE;
    req.intent.constraints.must_host_pinned = true;
    ggml_sycl::alloc_handle out{};
    TEST_ASSERT(!ggml_sycl::unified_alloc(req, &out) && out.ptr == nullptr,
                "unified_alloc must preallocate identity and refuse exhaustion before tier allocation");
    TEST_ASSERT(ggml_sycl::unified_cache_get_runtime_host_bytes() == host_used_before,
                "unified_alloc exhaustion must not leak or publish an allocation");

    TEST_PASS();
    return true;
}

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
    all_passed &= test_mem_handle_stable_identity();
    all_passed &= test_cache_entry_minted_retention_identity();
    // Must run last: it deliberately leaves the process-wide mint exhausted.
    // Device transitions are skipped only on a genuinely device-less host; the
    // identity-only tests above remain useful there.
    if (n_gpu_devices > 0) {
        try {
            sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
            all_passed &= test_retention_transitions_and_exhaustion(q);
        } catch (const sycl::exception & e) {
            fprintf(stderr, "[TEST] retention_transitions_and_exhaustion ... SKIPPED: %s\n", e.what());
        }
    } else {
        fprintf(stderr, "[TEST] retention_transitions_and_exhaustion ... SKIPPED: no GPU device\n");
    }

    fprintf(stderr, "-------------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed, %d skipped\n", g_tests_run, g_tests_passed, g_tests_skipped);

    if (!all_passed) {
        fprintf(stderr, "SOME TESTS FAILED\n");
        return 1;
    }
    fprintf(stderr, "ALL TESTS PASSED\n");
    return 0;
}
