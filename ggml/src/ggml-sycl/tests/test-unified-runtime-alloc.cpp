//
// Unified runtime allocator tests
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../ggml-sycl-test.hpp"
#include "../model-lifecycle.hpp"
#include "../unified-cache.hpp"
#include "../zone-sizing.hpp"
#include "pinned-host-block-test-common.hpp"
#include "sycl-spin-kernel.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

static int g_tests_run    = 0;
static int g_tests_passed = 0;

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

using namespace ggml_sycl;

static void enable_strict_mode_env() {
#if defined(_WIN32)
    (void) _putenv_s("GGML_SYCL_UNIFIED_ALLOC_STRICT", "1");
#else
    (void) setenv("GGML_SYCL_UNIFIED_ALLOC_STRICT", "1", 1);
#endif
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    (void) _putenv_s(name, value ? value : "");
#else
    if (value) {
        (void) setenv(name, value, 1);
    } else {
        (void) unsetenv(name);
    }
#endif
}

static bool reserve_allocate_success_registers_pointer(sycl::queue & q) {
    TEST_BEGIN("reserve_allocate_success_registers_pointer");
    alloc_request req{};
    req.queue                          = &q;
    req.size                           = 4096;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "unified_alloc failed");
    TEST_ASSERT(h.ptr != nullptr, "allocated pointer is null");

    alloc_metadata looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "lookup failed");
    TEST_ASSERT(looked.ptr == h.ptr, "lookup ptr mismatch");
    TEST_ASSERT(looked.size == h.size, "lookup size mismatch");

    TEST_ASSERT(unified_free(h), "free failed");
    TEST_PASS();
    return true;
}

static bool arena_registry_commit_failure_rolls_back(sycl::queue & q) {
    TEST_BEGIN("arena_registry_commit_failure_rolls_back");
    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    if (!cache->arena_active()) {
        TEST_PASS();
        return true;
    }

    alloc_request req{};
    req.queue                                      = &q;
    req.size                                       = 4096;
    req.intent.role                                = alloc_role::GRAPH_TMP;
    req.intent.category                            = runtime_category::GRAPH;
    req.intent.constraints.must_device             = true;
    req.intent.constraints.prefer_vram_zone         = vram_zone_id::RUNTIME;

    const size_t before = cache->zone_used(vram_zone_id::RUNTIME);
    unified_cache_test_fail_next_arena_registry_commit();
    alloc_handle failed{};
    TEST_ASSERT(!unified_alloc(req, &failed), "faulted registry publication unexpectedly succeeded");
    TEST_ASSERT(failed.ptr == nullptr, "faulted allocation returned a pointer");
    TEST_ASSERT(cache->zone_used(vram_zone_id::RUNTIME) == before,
                "registry insertion failure leaked TLSF bytes");
    // llama.cpp-ii25: the rolled-back row must not leave a range in the registry's containment index.
    TEST_ASSERT(allocation_registry_test_index_consistent(),
                "registry rollback left the containment index out of step");

    alloc_handle retry{};
    TEST_ASSERT(unified_alloc(req, &retry), "allocator did not recover after publication rollback");
    TEST_ASSERT(retry.alloc_id != 0, "retry omitted exact allocation identity");
    TEST_ASSERT(unified_free(retry), "retry cleanup failed");
    TEST_ASSERT(allocation_registry_test_index_consistent(),
                "freeing the arena row left the containment index out of step");
    TEST_PASS();
    return true;
}

static bool allocate_failure_rolls_back_budget(sycl::queue & q) {
    TEST_BEGIN("allocate_failure_rolls_back_budget");
    const int    device = 0;
    const size_t before = unified_cache_arena_non_weight_used(device);

    alloc_request req{};
    req.queue                          = &q;
    req.size                           = (size_t) 1 << 50;  // 1 PB-ish for deterministic fail on device alloc
    req.intent.role                    = alloc_role::GRAPH_TMP;
    req.intent.category                = runtime_category::GRAPH;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    const bool   ok = unified_alloc(req, &h);
    if (ok && h.ptr != nullptr) {
        // Unexpectedly succeeded, clean up and treat as pass.
        unified_free(h);
        TEST_PASS();
        return true;
    }
    const size_t after = unified_cache_arena_non_weight_used(device);
    TEST_ASSERT(before == after, "runtime bytes did not roll back after alloc failure");
    TEST_PASS();
    return true;
}

// A raw device request meets the device's physical bytes at unified_alloc's raw exit: the device total less the arena
// reservation, the raw rows held outside it and what was in use outside this process's accounting at the first cache.
// With an arena the plan placed the weights, so a raw request past those bytes is refused and evicts nothing. The case
// seeds one evictable cache entry whose eviction would free what the request is short of, sets the device total 1 MiB
// past what the process holds, and checks that a 4 MiB raw request is refused with that entry still cached, a 512 KiB
// one is served, and a 2 MiB request the RUNTIME zone serves never meets the gate.
static bool raw_gate_refuses_past_the_device_total(sycl::queue & q) {
    TEST_BEGIN("raw_gate_refuses_past_the_device_total");
    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    // Without an arena the gate may evict, so the no-eviction claim below means something only with one.
    TEST_ASSERT(cache->arena_active(), "no arena is active, so the arena's no-eviction rule cannot be exercised");
    constexpr size_t MiB = 1024 * 1024;

    alloc_request raw{};
    raw.queue                          = &q;
    raw.size                           = 4096;
    raw.intent.role                    = alloc_role::COMPUTE;
    raw.intent.category                = runtime_category::COMPUTE;
    raw.intent.constraints.must_device = true;

    // The device index the allocator resolves q to, read back from a small raw allocation.
    alloc_handle   probe{};
    alloc_metadata probe_meta{};
    TEST_ASSERT(unified_alloc(raw, &probe) && unified_lookup(probe.ptr, &probe_meta), "the probe allocation failed");
    TEST_ASSERT(unified_free(probe), "the probe free failed");
    const int device = probe_meta.device;
    TEST_ASSERT(cache->zone_available(vram_zone_id::RUNTIME) >= 2 * MiB,
                "the RUNTIME zone has under 2 MiB free, so the zone half cannot run");

    // One evictable entry outside the arena: a raw row the gate counts, in-use count 0, not pinned. It is the only
    // entry an eviction for the 4 MiB request could take, and taking it would free more than that request is short of.
    ggml_sycl_cache_id seed_id = {};
    seed_id.valid              = true;
    seed_id.model_id           = 0x7261772d67617465ull;  // an id no other case in this file uses
    seed_id.name_hash          = 1;
    seed_id.nbytes             = 4 * MiB;
    seed_id.type               = GGML_TYPE_F32;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        seed_id.ne[i] = 1;
    }
    std::vector<uint8_t> seed_src(4 * MiB, 0);
    bool                 needs_fill = false;
    void *               seed_ptr =
        cache->ensure_cached_alloc(seed_id, seed_src.data(), seed_src.size(), seed_src.size(),
                                   cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);
    TEST_ASSERT(seed_ptr != nullptr && cache->is_cached(seed_id, GGML_LAYOUT_AOS),
                "the evictable entry was not cached");

    const zone_raw_gate_inputs now = unified_cache_raw_gate_inputs(device, q);
    TEST_ASSERT(now.device_total != 0, "the device total is unknown, so the gate gates nothing");
    unified_cache_test_set_raw_gate_device_total(device, now.arena_bytes + now.raw_held + now.outside_accounting + MiB);

    // The cache keeps no eviction count: an eviction shows as the entry gone, the entry count and the cache's used
    // bytes down, and the sticky has_evictions flag raised (read only when it was clear before).
    const size_t entries_before   = cache->entry_count();
    const size_t used_before      = cache->used();
    const size_t weight_before    = cache->zone_used(vram_zone_id::WEIGHT);
    const bool   evictions_before = cache->has_evictions();
    raw.size                      = 4 * MiB;
    alloc_handle big{};
    const bool   big_ok        = unified_alloc(raw, &big);
    const bool   seed_kept     = cache->is_cached(seed_id, GGML_LAYOUT_AOS);
    const bool   nothing_moved = cache->entry_count() == entries_before && cache->used() == used_before &&
                               cache->zone_used(vram_zone_id::WEIGHT) == weight_before &&
                               (evictions_before || !cache->has_evictions());
    if (big_ok) {
        unified_free(big);
    }

    raw.size = 512 * 1024;
    alloc_handle small{};
    const bool   small_ok = unified_alloc(raw, &small) && small.ptr != nullptr && unified_free(small);

    // A request the RUNTIME zone serves is not a raw request: the gate's 1 MiB edge does not apply to it.
    alloc_request zoned                       = raw;
    zoned.size                                = 2 * MiB;
    zoned.intent.role                         = alloc_role::GRAPH_TMP;
    zoned.intent.category                     = runtime_category::GRAPH;
    zoned.intent.constraints.prefer_vram_zone = vram_zone_id::RUNTIME;
    alloc_handle in_zone{};
    const bool   zone_ok = unified_alloc(zoned, &in_zone) && in_zone.ptr != nullptr && unified_free(in_zone);

    unified_cache_test_set_raw_gate_device_total(device, 0);
    if (seed_kept) {
        cache->remove(seed_id, cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
        cache->evict(0);  // runs the deferred free of the removed entry
    }

    TEST_ASSERT(!big_ok && big.ptr == nullptr, "a raw request past the device's bytes was served");
    TEST_ASSERT(seed_kept, "the refusal evicted the cached entry an arena trip must leave alone");
    TEST_ASSERT(nothing_moved, "the refusal evicted cache entries or moved the WEIGHT zone");
    TEST_ASSERT(small_ok, "a raw request inside the device's bytes was refused");
    TEST_ASSERT(zone_ok, "a RUNTIME-zone request met the raw gate");
    TEST_PASS();
    return true;
}

static bool free_unknown_pointer_fails() {
    TEST_BEGIN("free_unknown_pointer_fails");
    int stack_value = 0;
    TEST_ASSERT(!unified_free_ptr(&stack_value, -1), "free unknown pointer should fail");
    TEST_PASS();
    return true;
}

static bool strict_unknown_free_fails() {
    TEST_BEGIN("strict_unknown_free_fails");
    int stack_value = 0;
    TEST_ASSERT(!unified_free_ptr(&stack_value, 0), "strict unknown free should fail");
    TEST_PASS();
    return true;
}

static bool double_free_fails(sycl::queue & q) {
    TEST_BEGIN("double_free_fails");
    alloc_request req{};
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::STAGING;
    req.intent.category                = runtime_category::STAGING;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    TEST_ASSERT(unified_free(h), "first free failed");
    TEST_ASSERT(!unified_free(h), "second free should fail");
    TEST_PASS();
    return true;
}

static bool lookup_returns_correct_metadata(sycl::queue & q) {
    TEST_BEGIN("lookup_returns_correct_metadata");
    alloc_request req{};
    req.queue                               = &q;
    req.size                                = 8192;
    req.intent.role                         = alloc_role::COMPUTE;
    req.intent.category                     = runtime_category::COMPUTE;
    req.intent.constraints.must_host_pinned = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    alloc_metadata looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "lookup failed");
    TEST_ASSERT(looked.tier == alloc_tier::HOST_PINNED, "tier mismatch");
    TEST_ASSERT(looked.role == alloc_role::COMPUTE, "role mismatch");
    TEST_ASSERT(looked.category == runtime_category::COMPUTE, "category mismatch");
    TEST_ASSERT(unified_free(h), "free failed");
    TEST_PASS();
    return true;
}

static bool cohort_prefers_weight_tier_for_compute(sycl::queue & q) {
    TEST_BEGIN("cohort_prefers_weight_tier_for_compute");
    alloc_request seed{};
    seed.queue                               = &q;
    seed.size                                = 4096;
    seed.intent.role                         = alloc_role::WEIGHT;
    seed.intent.category                     = runtime_category::OTHER;
    seed.intent.cohort_id                    = "test:cohort";
    seed.intent.constraints.must_host_pinned = true;

    alloc_handle seed_h{};
    TEST_ASSERT(unified_alloc(seed, &seed_h), "seed alloc failed");

    alloc_request req{};
    req.queue                                         = &q;
    req.size                                          = 2048;
    req.intent.role                                   = alloc_role::COMPUTE;
    req.intent.category                               = runtime_category::COMPUTE;
    req.intent.cohort_id                              = "test:cohort";
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    const alloc_tier tier                             = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::HOST_PINNED, "cohort policy did not preserve host tier");

    unified_free(seed_h);
    TEST_PASS();
    return true;
}

static bool hard_constraint_overrides_cohort(sycl::queue & q) {
    TEST_BEGIN("hard_constraint_overrides_cohort");
    alloc_request req{};
    req.queue                                         = &q;
    req.size                                          = 2048;
    req.intent.role                                   = alloc_role::COMPUTE;
    req.intent.category                               = runtime_category::COMPUTE;
    req.intent.cohort_id                              = "test:cohort";
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    req.intent.constraints.must_device                = true;
    const alloc_tier tier                             = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::DEVICE_VRAM, "must_device did not override cohort");
    TEST_PASS();
    return true;
}

static bool policy_never_selects_shared_usm(sycl::queue & q) {
    TEST_BEGIN("policy_never_selects_shared_usm");
    alloc_request req{};
    req.queue             = &q;
    req.size              = 1024;
    req.intent.role       = alloc_role::OTHER;
    req.intent.category   = runtime_category::OTHER;
    const alloc_tier tier = unified_select_tier(req);
    TEST_ASSERT(tier == alloc_tier::DEVICE_VRAM || tier == alloc_tier::HOST_PINNED, "unexpected tier selected");
    TEST_PASS();
    return true;
}

static bool strict_stale_handle_fails(sycl::queue & q) {
    TEST_BEGIN("strict_stale_handle_fails");
    alloc_request req{};
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    alloc_handle stale = h;
    TEST_ASSERT(unified_free(h), "free failed");
    TEST_ASSERT(!unified_free(stale), "stale handle free should fail");
    TEST_PASS();
    return true;
}

static bool strict_device_mismatch_fails(sycl::queue & q) {
    TEST_BEGIN("strict_device_mismatch_fails");
    alloc_request req{};
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    TEST_ASSERT(!unified_free_ptr(h.ptr, h.device + 1), "device mismatch free should fail");
    alloc_metadata looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "allocation should remain registered after mismatch");
    TEST_ASSERT(unified_free(h), "cleanup free failed");
    TEST_PASS();
    return true;
}

static bool scoped_unified_alloc_frees_on_scope_exit(sycl::queue & q) {
    TEST_BEGIN("scoped_unified_alloc_frees_on_scope_exit");
    alloc_request req{};
    req.queue                               = &q;
    req.size                                = 4096;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    void * ptr = nullptr;
    {
        scoped_unified_alloc scoped(req);
        TEST_ASSERT(scoped, "scoped allocation failed");
        ptr = scoped.get();
        TEST_ASSERT(ptr != nullptr, "scoped pointer null");
        alloc_metadata looked{};
        TEST_ASSERT(unified_lookup(ptr, &looked), "lookup should succeed while in scope");
    }
    alloc_metadata looked{};
    TEST_ASSERT(!unified_lookup(ptr, &looked), "lookup should fail after scope exit");
    TEST_PASS();
    return true;
}

static bool offload_pool_reuse_tracks_hit_miss(sycl::queue & q) {
    TEST_BEGIN("offload_pool_reuse_tracks_hit_miss");
    offload_buffer_pool_trim(-1);
    offload_stats_reset();

    offload_buffer_request req{};
    req.queue                                         = &q;
    req.device                                        = -1;
    req.size                                          = 4096;
    req.role                                          = offload_buffer_role::STAGING_SRC0;
    req.intent.role                                   = alloc_role::STAGING;
    req.intent.category                               = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned           = true;
    req.intent.constraints.prefer_same_tier_as_cohort = true;
    req.intent.cohort_id                              = "test:offload_pool";

    offload_buffer_lease a{};
    TEST_ASSERT(acquire_offload_buffer(req, &a), "first acquire failed");
    TEST_ASSERT(a.valid && a.handle.ptr != nullptr, "first lease invalid");
    TEST_ASSERT(release_offload_buffer(a), "first release failed");

    offload_buffer_lease b{};
    TEST_ASSERT(acquire_offload_buffer(req, &b), "second acquire failed");
    TEST_ASSERT(b.valid && b.handle.ptr != nullptr, "second lease invalid");
    TEST_ASSERT(release_offload_buffer(b), "second release failed");

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.pool_miss_count >= 1, "expected at least one pool miss");
    TEST_ASSERT(stats.pool_hit_count >= 1, "expected at least one pool hit");
    TEST_PASS();
    return true;
}

static bool offload_pool_stale_lease_fails(sycl::queue & q) {
    TEST_BEGIN("offload_pool_stale_lease_fails");
    offload_buffer_request req{};
    req.queue                               = &q;
    req.device                              = -1;
    req.size                                = 2048;
    req.role                                = offload_buffer_role::STAGING_DST;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease lease{};
    TEST_ASSERT(acquire_offload_buffer(req, &lease), "acquire failed");
    TEST_ASSERT(release_offload_buffer(lease), "release failed");
    TEST_ASSERT(!release_offload_buffer(lease), "stale lease release should fail");
    TEST_PASS();
    return true;
}

static bool offload_pool_trim_clears_released_entries(sycl::queue & q) {
    TEST_BEGIN("offload_pool_trim_clears_released_entries");
    offload_buffer_pool_trim(-1);
    offload_stats_reset();

    offload_buffer_request req{};
    req.queue                               = &q;
    req.device                              = -1;
    req.size                                = 1024;
    req.role                                = offload_buffer_role::STAGING_SRC1;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease lease{};
    TEST_ASSERT(acquire_offload_buffer(req, &lease), "first acquire failed");
    TEST_ASSERT(release_offload_buffer(lease), "release failed");
    offload_buffer_pool_trim(-1);

    offload_buffer_lease after_trim{};
    TEST_ASSERT(acquire_offload_buffer(req, &after_trim), "acquire after trim failed");
    TEST_ASSERT(release_offload_buffer(after_trim), "release after trim failed");

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.pool_miss_count >= 2, "expected miss after trim");
    TEST_PASS();
    return true;
}

static bool offload_pool_set_tensor_roles_reuse(sycl::queue & q) {
    TEST_BEGIN("offload_pool_set_tensor_roles_reuse");
    offload_buffer_pool_trim(-1);
    offload_stats_reset();

    offload_buffer_request stage_req{};
    stage_req.queue                               = &q;
    stage_req.device                              = -1;
    stage_req.size                                = 4096;
    stage_req.role                                = offload_buffer_role::SET_TENSOR_STAGE;
    stage_req.intent.role                         = alloc_role::STAGING;
    stage_req.intent.category                     = runtime_category::STAGING;
    stage_req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease stage_a{};
    TEST_ASSERT(acquire_offload_buffer(stage_req, &stage_a), "stage acquire A failed");
    TEST_ASSERT(release_offload_buffer(stage_a), "stage release A failed");
    offload_buffer_lease stage_b{};
    TEST_ASSERT(acquire_offload_buffer(stage_req, &stage_b), "stage acquire B failed");
    TEST_ASSERT(release_offload_buffer(stage_b), "stage release B failed");

    offload_buffer_request reorder_req = stage_req;
    reorder_req.role                   = offload_buffer_role::SET_TENSOR_REORDER;
    reorder_req.size                   = 8192;

    offload_buffer_lease reorder_a{};
    TEST_ASSERT(acquire_offload_buffer(reorder_req, &reorder_a), "reorder acquire A failed");
    TEST_ASSERT(release_offload_buffer(reorder_a), "reorder release A failed");

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.pool_hit_count >= 1, "expected pooled role reuse hit");
    TEST_ASSERT(stats.pool_miss_count >= 2, "expected misses for role bootstrap");
    TEST_PASS();
    return true;
}

static bool offload_wait_stats_split_tracks_forced_and_fallback() {
    TEST_BEGIN("offload_wait_stats_split_tracks_forced_and_fallback");
    offload_stats_reset();

    offload_stats_note_wait(false);
    offload_stats_note_wait(true);
    offload_stats_note_wait(false);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.wait_count == 3, "expected total wait_count to be 3");
    TEST_ASSERT(stats.wait_count_forced == 2, "expected wait_count_forced to be 2");
    TEST_ASSERT(stats.wait_count_fallback == 1, "expected wait_count_fallback to be 1");
    TEST_PASS();
    return true;
}

static bool offload_cross_domain_stats_split_by_phase() {
    TEST_BEGIN("offload_cross_domain_stats_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_cross_domain_transfer(0);
    offload_stats_note_cross_domain_transfer(128);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_cross_domain_transfer(256);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.cross_domain_transfer_count == 3, "expected three cross-domain boundaries");
    TEST_ASSERT(stats.cross_domain_transfer_count_pp == 2, "expected two PP cross-domain transfers");
    TEST_ASSERT(stats.cross_domain_transfer_count_tg == 1, "expected one TG cross-domain transfer");
    TEST_PASS();
    return true;
}

static bool offload_transfer_bytes_split_by_phase() {
    TEST_BEGIN("offload_transfer_bytes_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_transfer(true, 96);
    offload_stats_note_transfer(false, 48);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_transfer(true, 24);
    offload_stats_note_transfer(false, 12);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.transfer_bytes_h2d == 120, "unexpected total H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h == 60, "unexpected total D2H bytes");
    TEST_ASSERT(stats.transfer_bytes_h2d_pp == 96, "unexpected PP H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_h2d_tg == 24, "unexpected TG H2D bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h_pp == 48, "unexpected PP D2H bytes");
    TEST_ASSERT(stats.transfer_bytes_d2h_tg == 12, "unexpected TG D2H bytes");
    TEST_PASS();
    return true;
}

static bool offload_dispatch_counts_split_by_phase() {
    TEST_BEGIN("offload_dispatch_counts_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_dispatch(true, false);   // CPU
    offload_stats_note_dispatch(false, false);  // GPU
    offload_stats_note_dispatch(false, true);   // GPU island

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_dispatch(true, false);  // CPU
    offload_stats_note_dispatch(false, true);  // GPU island

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.dispatch_count_cpu == 2, "unexpected total CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu == 3, "unexpected total GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island == 2, "unexpected total GPU island dispatch count");
    TEST_ASSERT(stats.dispatch_count_cpu_pp == 1, "unexpected PP CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_cpu_tg == 1, "unexpected TG CPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_pp == 2, "unexpected PP GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_tg == 1, "unexpected TG GPU dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island_pp == 1, "unexpected PP GPU-island dispatch count");
    TEST_ASSERT(stats.dispatch_count_gpu_island_tg == 1, "unexpected TG GPU-island dispatch count");
    TEST_PASS();
    return true;
}

static bool offload_phase_roundtrip() {
    TEST_BEGIN("offload_phase_roundtrip");
    offload_stats_reset();
    TEST_ASSERT(offload_stats_phase() == offload_phase::UNKNOWN, "expected UNKNOWN after reset");
    offload_stats_set_phase(offload_phase::PP);
    TEST_ASSERT(offload_stats_phase() == offload_phase::PP, "expected PP phase");
    offload_stats_set_phase(offload_phase::TG);
    TEST_ASSERT(offload_stats_phase() == offload_phase::TG, "expected TG phase");
    TEST_PASS();
    return true;
}

static bool offload_transition_wait_stats_split_by_phase() {
    TEST_BEGIN("offload_transition_wait_stats_split_by_phase");
    offload_stats_reset();

    offload_stats_set_phase(offload_phase::PP);
    offload_stats_note_transition_wait(true);
    offload_stats_note_transition_wait(false);

    offload_stats_set_phase(offload_phase::TG);
    offload_stats_note_transition_wait(true);
    offload_stats_note_transition_wait(false);
    offload_stats_note_transition_wait(false);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.transition_wait_count == 2, "unexpected transition wait count");
    TEST_ASSERT(stats.transition_wait_count_pp == 1, "unexpected PP transition wait count");
    TEST_ASSERT(stats.transition_wait_count_tg == 1, "unexpected TG transition wait count");
    TEST_ASSERT(stats.transition_wait_elided_count == 3, "unexpected transition wait-elided count");
    TEST_ASSERT(stats.transition_wait_elided_count_pp == 1, "unexpected PP transition wait-elided count");
    TEST_ASSERT(stats.transition_wait_elided_count_tg == 2, "unexpected TG transition wait-elided count");
    TEST_PASS();
    return true;
}

static bool offload_host_alloc_stats_split_by_tag() {
    TEST_BEGIN("offload_host_alloc_stats_split_by_tag");
    offload_stats_reset();

    offload_stats_note_host_alloc("unified_alloc:host", 128);
    offload_stats_note_host_alloc("unified_alloc:host", 64);
    offload_stats_note_host_alloc("unified_cache:host_chunk", 256);
    offload_stats_note_host_alloc("host_malloc", 512);
    offload_stats_note_host_alloc("custom:other", 32);

    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.host_alloc_call_count == 5, "unexpected host_alloc_call_count");
    TEST_ASSERT(stats.host_alloc_bytes == 992, "unexpected host_alloc_bytes");
    TEST_ASSERT(stats.host_alloc_calls_unified_alloc_host == 2, "unexpected unified_alloc host calls");
    TEST_ASSERT(stats.host_alloc_bytes_unified_alloc_host == 192, "unexpected unified_alloc host bytes");
    TEST_ASSERT(stats.host_alloc_calls_unified_cache_host_chunk == 1, "unexpected host_chunk calls");
    TEST_ASSERT(stats.host_alloc_bytes_unified_cache_host_chunk == 256, "unexpected host_chunk bytes");
    TEST_ASSERT(stats.host_alloc_calls_host_malloc == 1, "unexpected host_malloc calls");
    TEST_ASSERT(stats.host_alloc_bytes_host_malloc == 512, "unexpected host_malloc bytes");
    TEST_ASSERT(stats.host_alloc_calls_other == 1, "unexpected other calls");
    TEST_ASSERT(stats.host_alloc_bytes_other == 32, "unexpected other bytes");
    TEST_PASS();
    return true;
}

static bool offload_raw_alloc_and_fallback_stats_are_counted() {
    TEST_BEGIN("offload_raw_alloc_and_fallback_stats_are_counted");
    offload_stats_reset();
    offload_stats_note_raw_device_alloc(4096);
    offload_stats_note_raw_device_alloc(8192);
    offload_stats_note_host_fallback_attempt(16384);
    const offload_stats_snapshot stats = offload_stats_get();
    TEST_ASSERT(stats.raw_device_alloc_call_count == 2, "unexpected raw device alloc calls");
    TEST_ASSERT(stats.raw_device_alloc_bytes == 12288, "unexpected raw device alloc bytes");
    TEST_ASSERT(stats.host_fallback_attempt_count == 1, "unexpected host fallback attempts");
    TEST_ASSERT(stats.host_fallback_attempt_bytes == 16384, "unexpected host fallback bytes");
    TEST_PASS();
    return true;
}

static bool direct_stage_host_fallback_counts_attempt(sycl::queue & q) {
    TEST_BEGIN("direct_stage_host_fallback_counts_attempt");
    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");

    // The fallback publishes a source in place only when a registry names it (design 5efe §4.8, §7 C6), so the
    // source is a pinned unified-cache block, not a raw sycl::malloc_host pointer. The block outlives the entry
    // dropped below.
    constexpr size_t src_size = 4096;
    pinned_host_block_test::block src_block(q, src_size, 0x5a);
    void * const                  src = src_block.data();

    static int key_tag;
    const ggml_sycl_cache_id key      = test_make_cache_id(&key_tag, 0xfeed);
    const size_t             dst_size = (size_t) 1 << 50;

    offload_stats_reset();
    const direct_stage_result result = cache->direct_stage_expert(key, src, src_size, dst_size, GGML_LAYOUT_SOA,
                                                                  nullptr, nullptr, &q);
    const offload_stats_snapshot stats = offload_stats_get();
    const size_t                 dropped = cache->drop_expert_entries_for_tensor_layout(
        std::vector<ggml_sycl_cache_id>{ key }, GGML_LAYOUT_AOS, "test-direct-stage-host-fallback-counts-attempt");

    TEST_ASSERT(result.ok && result.ptr == src, "direct-stage should fall back to the host-USM source");
    TEST_ASSERT(dropped == 1, "direct-stage fallback entry should be dropped before freeing source");
    TEST_ASSERT(stats.host_fallback_attempt_count == 1, "direct-stage fallback attempt was not counted once");
    TEST_ASSERT(stats.host_fallback_attempt_bytes == dst_size, "direct-stage fallback bytes should report dst size");
    TEST_PASS();
    return true;
}

static bool host_zone_contiguous_alloc_skips_chunk_tail(sycl::queue & q) {
    TEST_BEGIN("host_zone_contiguous_alloc_skips_chunk_tail");

    const char * old_chunk_mb = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    const bool   had_chunk_mb = old_chunk_mb != nullptr;
    std::string  saved_chunk_mb;
    if (had_chunk_mb) {
        saved_chunk_mb = old_chunk_mb;
    }
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");

    constexpr size_t  mib = 1024ull * 1024ull;
    pinned_chunk_pool pool(q, 128ull * mib);
    pool.configure_zones(12ull * mib, 2ull * mib, 40ull * mib, 2ull * mib);
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", had_chunk_mb ? saved_chunk_mb.c_str() : nullptr);

    TEST_ASSERT(pool.zones_configured(), "host zones were not configured");
    TEST_ASSERT(pool.zone_largest_free_block(host_zone_id::STAGING) >= 16ull * mib,
                "expected a full staging chunk after the partial chunk tail");

    void * ptr = pool.zone_alloc(host_zone_id::STAGING, 8ull * mib, pinned_chunk_pool::DEFAULT_ALIGNMENT);

    TEST_ASSERT(ptr != nullptr, "contiguous zone allocation should skip the partial chunk tail");
    pool.zone_free(host_zone_id::STAGING, ptr);

    TEST_PASS();
    return true;
}

// llama.cpp-glkg: pins the pinned_chunk_pool-level invariant the ordering
// fix in ggml-sycl.cpp relies on, and the boundary that fix deliberately
// does NOT cross. The root bug: configure_zones()'s capacity check
// (chunks_-only total_capacity) and its growth guard
// (total_allocated_ + chunk_size_ <= budget_, where total_allocated_ counts
// BOTH chunks_ AND runtime_chunks_) can disagree when runtime_chunks_ has
// already consumed the pool's budget through the pre-zone fallback path
// (ggml_backend_sycl_host_buffer_type_alloc_buffer's use_pinned_pool =
// host_zones_configured() = false branch) -- exactly what happened for 80+
// seconds on the reporting NAS box before S1-PRELOAD's lazy
// configure_host_zones_for_plan() call ever ran.
static bool host_zone_config_ordering_matters(sycl::queue & q) {
    TEST_BEGIN("host_zone_config_ordering_matters");

    const char * old_chunk_mb = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    const bool   had_chunk_mb = old_chunk_mb != nullptr;
    std::string  saved_chunk_mb;
    if (had_chunk_mb) {
        saved_chunk_mb = old_chunk_mb;
    }
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");

    constexpr size_t mib    = 1024ull * 1024ull;
    // 20 MiB budget, 16 MiB chunks: any single runtime allocation needing a
    // full fresh chunk leaves less than one more chunk's worth of budget
    // headroom, which is the exact shape of the master-era bug (there,
    // 13 x 2 GiB runtime chunks left 0.8 of 26.8 GB, less than one chunk).
    constexpr size_t budget = 20ull * mib;

    // Case 1: the invariant llama.cpp-glkg's ORDERING fix relies on --
    // configure_zones() called on a fresh pool, BEFORE any runtime-chunk
    // consumption, succeeds even for a footprint that will consume most of
    // the budget. This is what calling
    // ggml_sycl_configure_host_zones_for_plan() right after the placement
    // plan is finalized (ggml-sycl.cpp's
    // ggml_backend_sycl_set_tensor_inventory(), before create_tensor()
    // allocates any weight buffer) guarantees in production.
    {
        pinned_chunk_pool pool(q, budget);
        pool.configure_zones(4ull * mib, 1ull * mib, 2ull * mib, 1ull * mib);  // 8 MiB total footprint
        TEST_ASSERT(pool.zones_configured(), "fresh-pool configure_zones should succeed under budget");
    }

    // Case 2: the boundary this fix does NOT cross, and why not. If
    // runtime_chunks_ already committed most of the budget BEFORE
    // configure_zones() runs (the pre-zone-fallback scenario the bug
    // report's NAS trace shows), configure_zones() still reports the
    // shortfall and leaves zones disabled -- pinned_chunk_pool's own
    // capacity arithmetic was deliberately NOT changed to "adopt" bytes
    // already sitting in runtime_chunks_ into the zone system. Adoption was
    // considered and rejected: a runtime chunk's own tlsf_allocator already
    // reflects whatever is live-allocated within it, but configure_zones()
    // seeds a brand-new, EMPTY tlsf_allocator for each zone-chunk overlap
    // (see the zone_allocators_ construction loop in configure_zones()) --
    // reusing that chunk as zone-backing without also transplanting its
    // already-used ranges into the new zone-scoped allocator would let a
    // later zone_alloc() hand out bytes a live weight tensor already
    // occupies. llama.cpp-glkg's fix instead moves the
    // configure_host_zones_for_plan() CALL SITE earlier, so the standard
    // model-load path never reaches this state; this case pins that the
    // pool-level fallback the fix relies on not reaching has not quietly
    // changed underneath that ordering guarantee.
    {
        pinned_chunk_pool pool(q, budget);
        void *            pre_fill = pool.allocate_runtime(15ull * mib, pinned_chunk_pool::DEFAULT_ALIGNMENT);
        TEST_ASSERT(pre_fill != nullptr, "runtime pre-fill allocation failed");
        pool.configure_zones(4ull * mib, 1ull * mib, 2ull * mib, 1ull * mib);  // same 8 MiB footprint as case 1
        TEST_ASSERT(!pool.zones_configured(),
                    "documents current pinned_chunk_pool behavior: prior runtime consumption still blocks "
                    "configure_zones -- llama.cpp-glkg fixes ORDERING at the ggml-sycl.cpp call site, not "
                    "this pool-level arithmetic");
        pool.deallocate(pre_fill, 15ull * mib);
    }

    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", had_chunk_mb ? saved_chunk_mb.c_str() : nullptr);
    TEST_PASS();
    return true;
}

// llama.cpp-nsl3: the host WEIGHT zone must be able to hand out ONE
// contiguous allocation larger than a pinned chunk. zone_alloc() serves a
// contiguous pointer from a single chunk's TLSF, so growth sized to the
// shortfall and split into fixed chunk_size_ chunks adds aggregate capacity
// that no single-chunk allocation can consume: Gemma 4 E4B's 2856 MB
// host-placed per-layer embedding against 2048 MB chunks grew the zone by two
// more 2048 MB chunks and still failed model load. The runtime path
// (allocate_from_chunks -> grow_into) has always sized an oversize chunk as
// max(chunk_size_, align_up(size)); this pins that grow_zone() follows the
// same rule and reports the zone's added capacity from the actual chunk bytes.
static bool host_zone_grow_serves_oversize_contiguous_alloc(sycl::queue & q) {
    TEST_BEGIN("host_zone_grow_serves_oversize_contiguous_alloc");

    const char * old_chunk_mb = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    const bool   had_chunk_mb = old_chunk_mb != nullptr;
    std::string  saved_chunk_mb;
    if (had_chunk_mb) {
        saved_chunk_mb = old_chunk_mb;
    }
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");

    constexpr size_t mib    = 1024ull * 1024ull;
    constexpr size_t chunk  = 16ull * mib;
    constexpr size_t budget = 128ull * mib;
    const size_t     align  = pinned_chunk_pool::DEFAULT_ALIGNMENT;

    pinned_chunk_pool pool(q, budget);
    pool.configure_zones(4ull * mib, 1ull * mib, 2ull * mib, 1ull * mib);  // one 16 MiB chunk
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", had_chunk_mb ? saved_chunk_mb.c_str() : nullptr);
    TEST_ASSERT(pool.zones_configured(), "host zones were not configured");

    // Three chunks' worth in ONE contiguous request: the shape of the Gemma 4
    // failure scaled down (2856 MB against 2048 MB chunks).
    const size_t request = 3 * chunk;
    TEST_ASSERT(pool.zone_alloc(host_zone_id::WEIGHT, request, align) == nullptr,
                "a 48 MiB request must not fit the 4 MiB WEIGHT zone before growth");

    const size_t cap_before    = pool.zone_capacity(host_zone_id::WEIGHT);
    const size_t chunks_before = pool.chunk_count();
    // unified-cache.cpp's fragmentation path asks the pool's rule for the
    // growth: a fresh chunk must hold the whole allocation.
    const size_t need          = pinned_chunk_pool::zone_grow_request(request, align);
    TEST_ASSERT(pool.grow_zone(host_zone_id::WEIGHT, need), "grow_zone should succeed within a 128 MiB budget");
    const size_t used_before = pool.zone_used(host_zone_id::WEIGHT);

    void * ptr = pool.zone_alloc(host_zone_id::WEIGHT, request, align);
    TEST_ASSERT(ptr != nullptr, "zone_alloc of 3 chunks after grow_zone should return a contiguous pointer");

    const size_t cap_after    = pool.zone_capacity(host_zone_id::WEIGHT);
    const size_t chunks_after = pool.chunk_count();
    const size_t grown_usable = pinned_chunk_pool::chunk_footprint_for(chunk, need).usable;
    TEST_ASSERT(chunks_after == chunks_before + 1, "oversize growth should add exactly one chunk");
    TEST_ASSERT(cap_after - cap_before == grown_usable,
                "zone_capacity should grow by the actual bytes of the oversize chunk");
    // The allocation takes its TLSF block, the request rounded to the grain, which the new chunk holds: the accounted
    // bytes are at least the request and at most the new chunk.
    const size_t used_delta = pool.zone_used(host_zone_id::WEIGHT) - used_before;
    TEST_ASSERT(used_delta >= request && used_delta <= grown_usable,
                "the oversize allocation should be accounted within the new chunk's bytes");

    // Budget is checked against the actual bytes: a growth that does not fit
    // the remaining budget is refused without adding any chunk.
    TEST_ASSERT(!pool.grow_zone(host_zone_id::WEIGHT, 100ull * mib),
                "growth beyond the remaining budget should be refused");
    TEST_ASSERT(pool.chunk_count() == chunks_after, "a refused growth must not add a chunk");
    TEST_ASSERT(pool.zone_capacity(host_zone_id::WEIGHT) == cap_after, "a refused growth must not change capacity");

    pool.zone_free(host_zone_id::WEIGHT, ptr);
    TEST_PASS();
    return true;
}

// The same growth for a request whose size is not a multiple of the TLSF grain: zone_alloc rounds it up to the grain,
// so the chunk grown for it must hold the rounded block, not only the request plus the base alignment. A size 32 B past
// three chunks is the shape a SYCL_Host buffer takes (ggml-alloc pads its tensors to 32 B).
static bool host_zone_grow_serves_odd_residue_oversize_alloc(sycl::queue & q) {
    TEST_BEGIN("host_zone_grow_serves_odd_residue_oversize_alloc");

    const char * old_chunk_mb = std::getenv("GGML_SYCL_PINNED_CHUNK_MB");
    const bool   had_chunk_mb = old_chunk_mb != nullptr;
    std::string  saved_chunk_mb;
    if (had_chunk_mb) {
        saved_chunk_mb = old_chunk_mb;
    }
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");

    constexpr size_t mib    = 1024ull * 1024ull;
    constexpr size_t chunk  = 16ull * mib;
    constexpr size_t budget = 128ull * mib;
    const size_t     align  = pinned_chunk_pool::DEFAULT_ALIGNMENT;

    pinned_chunk_pool pool(q, budget);
    pool.configure_zones(4ull * mib, 1ull * mib, 2ull * mib, 1ull * mib);  // one 16 MiB chunk
    set_env_var("GGML_SYCL_PINNED_CHUNK_MB", had_chunk_mb ? saved_chunk_mb.c_str() : nullptr);
    TEST_ASSERT(pool.zones_configured(), "host zones were not configured");

    const size_t request = 3 * chunk + 32;
    TEST_ASSERT(pool.zone_alloc(host_zone_id::WEIGHT, request, align) == nullptr,
                "a request past three chunks must not fit the 4 MiB WEIGHT zone before growth");
    const size_t cap_before = pool.zone_capacity(host_zone_id::WEIGHT);
    const size_t need       = pinned_chunk_pool::zone_grow_request(request, align);
    TEST_ASSERT(pool.grow_zone(host_zone_id::WEIGHT, need), "grow_zone should succeed within a 128 MiB budget");
    TEST_ASSERT(pool.zone_capacity(host_zone_id::WEIGHT) - cap_before ==
                    pinned_chunk_pool::chunk_footprint_for(chunk, need).usable,
                "zone_capacity should grow by the usable bytes of the chunk grown for the request");

    void * ptr = pool.zone_alloc(host_zone_id::WEIGHT, request, align);
    TEST_ASSERT(ptr != nullptr, "the chunk grown for a request off the TLSF grain should serve it");

    pool.zone_free(host_zone_id::WEIGHT, ptr);
    TEST_PASS();
    return true;
}

// Run only via --case host_inventory_initializes_zones, in a fresh process.
// Metadata describes complete gate/up/down groups (one MiB per role) whose
// aggregate exceeds the actual shared arena by 64 MiB. No GGUF or weight
// payload is loaded. Empty/all-device plans
// are failures, not a vacuous ordering PASS. See the arithmetic below: 1% of
// the B50 is LESS than external headroom and aborts before this boundary.
static bool host_inventory_initializes_zones(sycl::queue & q) {
    TEST_BEGIN("host_inventory_initializes_zones");
    constexpr size_t mib = 1024ull * 1024ull;
    const auto device = q.get_device();
    const size_t device_bytes = device.get_info<sycl::info::device::global_mem_size>();
    TEST_ASSERT(device.is_gpu() && !device.get_info<sycl::info::device::host_unified_memory>() &&
                    device_bytes >= 8192 * mib && device_bytes <= 20480 * mib,
                "fixture requires an 8-20 GiB discrete GPU (lead-selected B50), not the iGPU");
    // Keep the real ONEDNN policy, but bound unused compute/runtime zones
    // to 16 MiB each (this fixture submits no inference). Conservative tail:
    // 256 ONEDNN + 16 SCRATCH + 16 RUNTIME + 16 minimum shared = 304 MiB.
    // Add 16 MiB margin and round the authority's result down to the arena's
    // 2-MiB granularity. Choose the FIRST percentage meeting that bound via
    // the production authority, which subtracts external headroom exactly
    // once. At the observed B50 total of 16304 MiB this chooses 12%, yielding
    // 326 MiB (not the former 1% -> zero). The ONEDNN quarter-budget clamp
    // can only reduce the conservative tail, not invalidate its bound.
    constexpr size_t min_arena = 320 * mib;
    constexpr size_t max_arena = 512 * mib;
    constexpr size_t arena_alignment = 2 * mib;
    int budget_pct = 0;
    for (int pct = 1; pct <= 100; ++pct) {
        const auto authority = compute_vram_budget_authority(false, device_bytes, device_bytes, device_bytes, pct);
        const size_t rounded = authority.budget_bytes / arena_alignment * arena_alignment;
        if (rounded >= min_arena) {
            TEST_ASSERT(rounded <= max_arena, "minimum usable arena exceeds fixture's 512-MiB device bound");
            budget_pct = pct;
            break;
        }
    }
    TEST_ASSERT(budget_pct != 0, "no usable bounded arena budget");
    const std::string pct_text = std::to_string(budget_pct);
    set_env_var("GGML_SYCL_VRAM_BUDGET_PCT", pct_text.c_str());
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_sycl_init(0), ggml_backend_free);
    TEST_ASSERT(backend != nullptr, "SYCL backend initialization failed");
    auto * cache = get_unified_cache_for_device(0);
    TEST_ASSERT(cache != nullptr, "backend cache unavailable");
    TEST_ASSERT(cache->get_queue().get_device() == q.get_device(), "fixture queue/backend device mismatch");
    TEST_ASSERT(!cache->host_zones_configured(), "fixture requires a fresh cache without host zones");
    TEST_ASSERT(cache->arena_active() && cache->arena_total_size() >= min_arena &&
                    cache->arena_total_size() <= max_arena,
                "actual free-VRAM authority did not yield the bounded usable arena; do not call inventory");
    TEST_ASSERT(cache->pinned_pool_budget() >= 1536 * mib && cache->pinned_pool_budget() <= 4096 * mib,
                "fixture requires a 1.5-4 GiB host pool budget; use the documented local host");
    const size_t shared_bytes = cache->zone_capacity(vram_zone_id::WEIGHT);
    constexpr size_t group_bytes = 3 * mib;
    const size_t n_experts = (shared_bytes + 64 * mib + group_bytes - 1) / group_bytes;
    TEST_ASSERT(3 * n_experts <= 512, "synthetic inventory would exceed 512-MiB metadata bound");
    fprintf(stderr, "fixture pct=%d arena=%zu shared=%zu experts=%zu external_headroom=%zu\n",
            budget_pct, cache->arena_total_size(), shared_bytes, n_experts, cache->external_headroom());

    // CONTROL requires a complete logical role set, not the gate-only
    // descriptor that produced metadata_mismatch=1 in the f6cae004 run.
    const char * names[] = {
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
    };
    ggml_sycl_tensor_info tensors[3]{};
    for (size_t i = 0; i < 3; ++i) {
        tensors[i].name  = names[i];
        tensors[i].type  = GGML_TYPE_F32;
        tensors[i].ne[0] = 512;
        tensors[i].ne[1] = 512;
        tensors[i].ne[2] = static_cast<int64_t>(n_experts);
        tensors[i].ne[3] = 1;
        tensors[i].size  = n_experts * mib;
    }
    ggml_sycl_tensor_inventory inventory{};
    inventory.tensors      = tensors;
    inventory.count        = 3;
    inventory.total_size   = group_bytes * n_experts;
    inventory.n_expert     = static_cast<int>(n_experts);
    inventory.n_expert_used = 1;
    inventory.n_layer      = 1;
    inventory.n_ctx        = 16;
    inventory.n_ubatch     = 1;

    // The production loader binds a transaction and stages inventory through
    // this public wrapper. Naked early/late setters compute a local snapshot
    // but do not stage it without the wrapper's load-effect authority. Cache
    // snapshots are publication diagnostics, NOT the early load candidate.
    const auto admission_before = lifecycle::global_registry().admission_diagnostics();
    TEST_ASSERT(admission_before.active_txn == 0 && admission_before.models == 0 &&
                    !admission_before.shutdown_reserved && !admission_before.shutdown_completed,
                "fixture requires an idle, empty, non-shutdown lifecycle registry");
    struct load_scope {
        lifecycle::admission_diagnostics_snapshot before;
        ggml_sycl_load_txn txn{};
        bool active = false;

        bool abort() noexcept {
            if (!active) {
                fprintf(stderr, "FAILED: host inventory cleanup called without active transaction\n");
                return false;
            }
            active = false;
            const auto rc = ggml_backend_sycl_model_load_end(txn, false, nullptr);
            const bool candidate_absent = !lifecycle_find_candidate_placement_plan(txn.id);
            const auto after = lifecycle::global_registry().admission_diagnostics();
            // end(false) rolls back with reason MISSING_SUCCESS; ABORTED is
            // the terminal phase, not this API's result. EFFECT_FAILED and
            // every other result remain failures, even if counts look empty.
            const bool clean = rc == GGML_SYCL_LIFECYCLE_MISSING_SUCCESS && candidate_absent &&
                               after.active_txn == 0 && after.models == before.models &&
                               after.shutdown_reserved == before.shutdown_reserved &&
                               after.shutdown_completed == before.shutdown_completed;
            fprintf(stderr, "%s host inventory cleanup: result=%d candidate_absent=%d active=%llu models=%llu "
                            "baseline_models=%llu\n",
                    clean ? "PASS:" : "FAILED:", static_cast<int>(rc), static_cast<int>(candidate_absent),
                    static_cast<unsigned long long>(after.active_txn), static_cast<unsigned long long>(after.models),
                    static_cast<unsigned long long>(before.models));
            return clean;
        }

        ~load_scope() {
            if (active) {
                // Assertion/exception paths already fail the case; abort()
                // still prints the full semantic receipt and any failure.
                (void) abort();
            }
        }
    } load{ admission_before };
    const auto begin_rc = ggml_backend_sycl_model_load_begin(&load.txn);
    load.active = begin_rc == GGML_SYCL_LIFECYCLE_OK;
    TEST_ASSERT(load.active, "public model-load begin failed");
    const auto admission_during = lifecycle::global_registry().admission_diagnostics();
    // begin_outer reserves a slot and inserts txns_, NOT models_. models_
    // gains its transient row in prepare_end and loses it on clean rollback.
    // Prove this observer sees the public begin before trusting its later
    // zero: an unrelated/empty registry cannot satisfy active_txn==txn.id.
    TEST_ASSERT(load.txn.id != 0 && admission_during.active_txn == load.txn.id &&
                    admission_during.models == admission_before.models,
                "public begin not visible in exact registry, or unexpected model admission change");
    const auto begin_candidate = lifecycle_find_candidate_placement_plan(load.txn.id);
    TEST_ASSERT(begin_candidate && begin_candidate->load_txn_id == load.txn.id && begin_candidate->explicit_no_plan,
                "public begin did not expose its exact initial candidate");
    TEST_ASSERT(ggml_backend_sycl_stage_load_inventory_plan(&inventory, nullptr, true) == GGML_SYCL_LIFECYCLE_OK,
                "public early inventory staging failed");
    const auto snapshot = lifecycle_find_candidate_placement_plan(load.txn.id);
    TEST_ASSERT(snapshot && snapshot->load_txn_id == load.txn.id && !snapshot->explicit_no_plan && snapshot->plan,
                "public early inventory did not stage the exact transaction candidate");
    const auto & plan = *snapshot->plan;
    TEST_ASSERT(plan.entries.size() == 3 * n_experts, "synthetic inventory did not plan every expert role");
    TEST_ASSERT(plan.base_context_control_layout.valid, "complete descriptors produced invalid CONTROL layout");
    TEST_ASSERT(plan.weight_host_bytes > 0, "synthetic inventory produced no host placement");
    TEST_ASSERT(!cache->host_zones_configured(), "early inventory unexpectedly configured host zones");
    // Match the late configure helper, including its KV/scratch floors and
    // the separate runtime-chunk preallocation. Ceil each aggregate to a
    // 16-MiB chunk; allow one further chunk for aligned zone boundaries.
    const size_t host_scratch = std::max(plan.host_zone_scratch_bytes,
        cache->onednn_weights_scratch_size() + cache->onednn_activations_scratch_size() +
        plan.max_copied_tensor_bytes + 32 * mib);
    const size_t host_zones = plan.host_zone_weight_bytes + std::max(plan.host_zone_kv_bytes, 64 * mib) +
                             plan.host_zone_staging_bytes + host_scratch;
    const size_t host_runtime = plan.onednn_scratchpad_bytes + plan.dma_staging_pool_bytes;
    TEST_ASSERT(host_zones <= 1536 * mib && host_runtime <= 128 * mib,
                "planned host footprint exceeds fixture bound; do not call late inventory");
    constexpr size_t chunk = 16 * mib;
    const size_t host_committed_bound = (host_zones + chunk - 1) / chunk * chunk +
                                       (host_runtime + chunk - 1) / chunk * chunk + chunk;
    TEST_ASSERT(host_committed_bound <= 1680 * mib &&
                    cache->pinned_pool_committed() + host_committed_bound <= cache->pinned_pool_budget(),
                "host provisioning exceeds fixture cap or pool budget");
    size_t planned_runtime = 0;
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(0, &planned_runtime) &&
                    planned_runtime <= cache->zone_capacity(vram_zone_id::RUNTIME),
                "inventory would grow mandatory runtime tail beyond the bounded arena");
    TEST_ASSERT(unified_cache_get_planned_onednn_scratchpad_bytes(0) <= 256 * mib,
                "inventory would exceed the conservative ONEDNN tail bound");
    fprintf(stderr, "host_zones=%zu host_runtime=%zu committed_bound=%zu\n",
            host_zones, host_runtime, host_committed_bound);
    // Now cross the real late-inventory boundary. The headroom its
    // [VRAM-BUDGET] lines print (base_mem/4 when weights exceed shared
    // capacity) is a figure in those lines, not a second budget subtraction.
    TEST_ASSERT(ggml_backend_sycl_stage_load_inventory_plan(&inventory, nullptr, false) == GGML_SYCL_LIFECYCLE_OK,
                "public late inventory staging failed");
    const auto late_snapshot = lifecycle_find_candidate_placement_plan(load.txn.id);
    TEST_ASSERT(late_snapshot && late_snapshot->load_txn_id == load.txn.id && !late_snapshot->explicit_no_plan &&
                    late_snapshot->plan && late_snapshot->plan->entries.size() == 3 * n_experts &&
                    late_snapshot->plan->weight_host_bytes > 0,
                "public late inventory did not retain a nonempty exact host candidate");
    fprintf(stderr, "entries=%zu host_weights=%zu committed=%zu budget=%zu\n",
            late_snapshot->plan->entries.size(), late_snapshot->plan->weight_host_bytes,
            cache->pinned_pool_committed(), cache->pinned_pool_budget());
    // Expected historical RED at 7b03ea03f: planning returns before the only
    // zone-configuration calls (S1-PRELOAD). Do not allocate first: that would
    // consume runtime chunks and obscure the ordering assertion's cause.
    TEST_ASSERT(cache->host_zones_configured(), "HOST_ORDER_RED: nonempty host plan returned before host zone setup");

    auto * buft = ggml_backend_sycl_host_buffer_type_for_device(ggml_backend_get_device(backend.get()));
    TEST_ASSERT(buft != nullptr, "exact-device host buffer type unavailable");
    constexpr size_t request_bytes = 993280;  // reporter's -c 4096 output request
    TEST_ASSERT(ggml_backend_buft_get_max_size(buft) >= request_bytes, "host buft cannot advertise reporter request");
    const size_t before = cache->host_zone_used(host_zone_id::WEIGHT);
    {
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_buft_alloc_buffer(buft, request_bytes), ggml_backend_buffer_free);
        TEST_ASSERT(buffer != nullptr, "public host buffer allocation failed after inventory");
        alloc_metadata metadata{};
        TEST_ASSERT(unified_lookup(ggml_backend_buffer_get_base(buffer.get()), &metadata),
                    "host buffer lacks allocation metadata");
        TEST_ASSERT(metadata.zone_managed && metadata.host_zone == host_zone_id::WEIGHT,
                    "public host buffer did not route to accounted WEIGHT zone");
        TEST_ASSERT(cache->host_zone_used(host_zone_id::WEIGHT) >= before + request_bytes,
                    "host buffer allocation was not charged to WEIGHT");
    }
    TEST_ASSERT(cache->host_zone_used(host_zone_id::WEIGHT) == before, "host buffer release leaked WEIGHT usage");
    // This allocation happens DURING loading: load-side routing coverage,
    // NOT a max-size regression witness (the chunk floor can mask it), and
    // NOT proof of post-load context headroom or the 64-MiB reserve. Never
    // finish successfully just to publish a cache snapshot: that runs S1.
    // The inner buffer owner is gone before abort on all paths, including
    // assertion returns and exceptions. Abort skips preload; check its result
    // explicitly on success, and let the guard handle failing paths.
    TEST_ASSERT(load.abort(), "public model-load abort semantic cleanup receipt failed");
    TEST_PASS();
    return true;
}

static std::string shell_quote(const char * value) {
#if defined(_WIN32)
    return std::string("\"") + value + "\"";
#else
    std::string quoted("'");
    for (const char * p = value; *p; ++p) {
        if (*p == '\'') {
            quoted += "'\\''";
        } else {
            quoted += *p;
        }
    }
    quoted += '\'';
    return quoted;
#endif
}

static bool global_cache_static_destruction_exits_cleanly(const char * self) {
    TEST_BEGIN("global_cache_static_destruction_exits_cleanly");
    const std::string command = shell_quote(self) + " --static-destruction-child";
    TEST_ASSERT(std::system(command.c_str()) == 0, "cache subprocess did not exit cleanly");
    TEST_PASS();
    return true;
}

static release_attempt publication_race_release_backend(const alloc_metadata &, void * context) noexcept {
    static_cast<std::atomic<unsigned> *>(context)->fetch_add(1, std::memory_order_release);
    return { release_attempt_status::RELEASED };
}

static bool accepted_control_publication_is_atomic_with_close_snapshot() {
    TEST_BEGIN("B70_accepted_control_publication_is_atomic_with_close_snapshot");
    static int marker;
    alloc_metadata metadata{};
    metadata.ptr = &marker;
    metadata.size = sizeof(marker);
    metadata.device = 7;
    metadata.id = 0x7070;
    metadata.role = alloc_role::COMPUTE;
    metadata.category = runtime_category::COMPUTE;

    std::atomic<unsigned> releases{ 0 };
    auto seed = allocation_owner_test_create(metadata, publication_race_release_backend, &releases,
                                             allocation_error::CONTROL_ALLOCATION_FAILED);
    TEST_ASSERT(seed.coordinator != nullptr, "test coordinator creation failed");

    allocation_result published;
    allocation_owner_test_pause_control_publication(true);
    std::thread publisher([&] {
        published = allocation_owner_test_create_on_coordinator(metadata, seed.coordinator);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!allocation_owner_test_control_publication_reached() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!allocation_owner_test_control_publication_reached()) {
        allocation_owner_test_pause_control_publication(false);
        publisher.join();
        TEST_FAIL("publication did not reach deterministic admission barrier");
    }
    std::thread closer([&] { allocation_coordinator_test_close(seed.coordinator); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    allocation_owner_test_pause_control_publication(false);
    publisher.join();
    closer.join();

    TEST_ASSERT(published, "accepted registration lost the close race");
    const auto controls = seed.coordinator->snapshot_controls();
    TEST_ASSERT(controls.size() == 1, "snapshot missed the accepted registration");
    TEST_ASSERT(controls[0].metadata == metadata, "snapshot observed partial control metadata");
    TEST_ASSERT(controls[0].ownership_class == allocation_control_class::EXTERNAL_EXACT,
                "snapshot observed an uninitialized ownership classification");
    TEST_ASSERT(published.owner.reset().released(), "published control cleanup failed");
    TEST_ASSERT(releases.load(std::memory_order_acquire) == 1, "release backend was not invoked exactly once");
    TEST_PASS();
    return true;
}

static bool retained_pinned_suballocation_refuses_preteardown(sycl::queue & q) {
    TEST_BEGIN("B50_B70_retained_pinned_suballocation_refuses_preteardown");
    constexpr size_t mib = 1024ull * 1024ull;
    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    if (!cache->host_zones_configured()) cache->configure_host_zones(2ull * mib, 2ull * mib, 4ull * mib, 2ull * mib);
    TEST_ASSERT(cache->host_zones_configured(), "host zones unavailable for pinned-owner fixture");

    uint64_t census_before[3]{};
    TEST_ASSERT(unified_cache_shutdown_allocation_control_census_for_test(census_before),
                "allocation-control census unavailable");

    alloc_request req{};
    req.queue = &q;
    req.device = 0;  // ONEAPI_DEVICE_SELECTOR remaps the selected B50/B70 to runtime device 0.
    req.size = 4096;
    req.intent.role = alloc_role::STAGING;
    req.intent.category = runtime_category::STAGING;
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool = true;

    const size_t used_before_failure = cache->host_zone_used(host_zone_id::STAGING);
    allocation_owner_test_fail_next_control_allocations(1);
    allocation_result failed = unified_allocate_owner(req);
    TEST_ASSERT(!failed && failed.error == allocation_error::CONTROL_ALLOCATION_FAILED,
                "injected control allocation failure unexpectedly succeeded");
    uint64_t census_after_failure[3]{};
    TEST_ASSERT(unified_cache_shutdown_allocation_control_census_for_test(census_after_failure) &&
                    census_after_failure[0] == census_before[0] && census_after_failure[1] == census_before[1] &&
                    census_after_failure[2] == census_before[2],
                "failed allocation leaked an allocation control");
    TEST_ASSERT(cache->host_zone_used(host_zone_id::STAGING) == used_before_failure,
                "failed allocation leaked a pinned-pool registry row or TLSF block");

    mem_handle retained = unified_allocate(req);
    const int device = req.device;
    resolved_ptr resolved = retained.resolve();
    TEST_ASSERT(retained.valid(), "pinned suballocation owner construction failed");
    TEST_ASSERT(resolved.ptr != nullptr, "pinned suballocation did not resolve");
    TEST_ASSERT(!resolved.on_device, "pinned suballocation resolved as device memory");
    auto coordinator = allocation_coordinator_test_lookup(device);
    TEST_ASSERT(coordinator != nullptr, "allocation coordinator unavailable");
    const auto live = coordinator->snapshot_controls();
    TEST_ASSERT(std::any_of(live.begin(), live.end(), [](const allocation_control_snapshot & control) {
                    return control.ownership_class == allocation_control_class::CACHE_BACKING;
                }),
                "cache backing owner was not classified at its creation route");
    const auto retained_control = std::find_if(live.begin(), live.end(), [&](const allocation_control_snapshot & c) {
        return c.metadata.ptr == resolved.ptr;
    });
    TEST_ASSERT(retained_control != live.end() &&
                    retained_control->ownership_class == allocation_control_class::CACHE_SUBALLOCATION,
                "pinned pool slice was not classified as a cache suballocation");
    const size_t capacity_before = cache->host_zone_capacity(host_zone_id::STAGING);
    std::memset(resolved.ptr, 0x5a, req.size);

    TEST_ASSERT(!shutdown_unified_cache(), "shutdown destroyed caches beneath a retained pinned owner");
    TEST_ASSERT(get_unified_cache_for_device(device) == cache, "refused shutdown detached the global cache");
    TEST_ASSERT(cache->host_zones_configured() &&
                    cache->host_zone_capacity(host_zone_id::STAGING) == capacity_before,
                "refused shutdown destroyed or reconfigured the pinned pool");
    TEST_ASSERT(static_cast<unsigned char *>(resolved.ptr)[0] == 0x5a,
                "pinned backing became unusable after refused shutdown");

    retained = {};
    TEST_ASSERT(shutdown_unified_cache(), "shutdown retry failed after pinned owner release");
    TEST_PASS();
    return true;
}

static bool explicit_global_cache_shutdown_is_clean() {
    TEST_BEGIN("explicit_global_cache_shutdown_is_clean");
    TEST_ASSERT(shutdown_unified_cache(), "explicit global cache shutdown failed");
    TEST_ASSERT(unified_cache_shutdown_state_clean(), "global cache retained owners after explicit shutdown");
    uint64_t controls[3]{};
    TEST_ASSERT(unified_cache_shutdown_allocation_control_census_for_test(controls),
                "allocation-control census failed after explicit shutdown");
    TEST_ASSERT(controls[0] == 0 && controls[1] == 0 && controls[2] == 0,
                "explicit shutdown did not detach a zero-control coordinator map");
    TEST_PASS();
    return true;
}

#if GGML_SYCL_DNNL
// llama.cpp-me60 STEP A (isolation case): while investigating why the RED
// test below failed with an unpredicted message, a SEPARATE, earlier
// defect was found: onednn_graph_scratch_free()'s DIRECT branch moves
// (does not release) the parked buffer's own EXTERNAL_EXACT allocation
// control into onednn_graph_scratch_reuse_pool_, and
// snapshot_allocation_controls(..., preteardown=true) USED TO run at the
// very start of shutdown_unified_cache() -- strictly before any cache's
// shutdown_resources() (and therefore before
// onednn_graph_scratch_clear_pool_locked() ever ran) -- before the
// pre-census pool drain+reclaim pass was inserted ahead of it
// (llama.cpp-me60's pre-census pass, this case's own STEP A defect).
// That pre-teardown census refused on any live non-CACHE_BACKING control,
// so the parked buffer itself tripped it before the LATER registry sweep
// F1 targets was ever reached. Traced (by direct code read, not built) to
// predate llama.cpp-c6ah entirely -- the same structure and ordering
// already existed at 0ef69a3d3 (pre-c6ah master); it is from
// llama.cpp-0oxf, which introduced the DIRECT reuse pool.
//
// This case isolates the two: reclaim the pool explicitly, through the
// same public entry point ggml_backend_sycl_set_runtime_context() already
// uses in production (ggml-sycl.cpp), BEFORE calling shutdown -- releasing the
// parked buffer's own control ahead of the pre-teardown census so it
// cannot be what refuses this time. Whatever shutdown_unified_cache()
// returns here says whether the FLAG SLAB's own registry row (what F1 as
// filed actually targets) independently trips the LATER
// unified_cache_shutdown_retryable_postconditions_clean() sweep, once the
// pool-census defect is not in the way.
//
// RUNS ONLY VIA ITS OWN ctest REGISTRATION
// (sycl-runtime-alloc-flag-slab-survives-after-pool-reclaim, `--case
// <name>`), in a fresh process, with nothing run before it -- NOT from
// main()'s default (no-argument) run. Two hardware findings established
// why, in order:
//
// (1) A REFUSED shutdown_unified_cache() call is not retryable-safe here
// -- the sibling case's own refused shutdown (by design -- see that
// case's own comment) left ITS cache PARTIALLY torn down: its parked
// buffer's control kept the cache registered in the per-device cache map, but a
// subsequent onednn_graph_scratch_alloc() on that same cache resolved
// off-device and hit GGML_ABORT at
// onednn_graph_scratch_alloc_direct_locked(). So this case cannot simply
// run BEFORE the sibling case in the same process either, contrary to an
// earlier version of this comment (main() briefly tried exactly that
// ordering, with this case forcing its own clean cache by calling
// shutdown_unified_cache() and ignoring the result).
//
// (2) Running ANY earlier case in this process that itself completes a
// successful shutdown_unified_cache() call ALSO breaks this one --
// verified on hardware (build-me60-4/5): g_sycl_shutting_down is left
// `true` at the end of every SUCCESSFUL shutdown_unified_cache() call
// (its own store(true) at the very end), and is reset to `false` only
// MID-WAY through the NEXT shutdown_unified_cache() call -- AFTER that
// call's own shutdown_resources() loop has already run using the STALE
// `true` value from the PREVIOUS call. get_unified_cache(q) lazily
// recreating a cache after such a prior shutdown does NOT reset the flag
// (only the production reactivation hooks -- prepare_unified_cache_for_module_use(),
// called from ggml_backend_sycl_commit_reactivate() and
// ggml_backend_sycl_reg() -- do that; this file's plain get_unified_cache(q)
// bypasses that whole protocol). So THIS case's own fresh cache, though
// genuinely newly-constructed, still sees g_sycl_shutting_down == true
// when ITS shutdown_resources() call runs, and takes the "SYCL might
// already be gone" ABANDON branch instead of the normal release path --
// which leaves the flag slab's registry row behind exactly the way F1
// itself does, but for an entirely different, TEST-ONLY reason (the
// abandon branch is correct production behavior for a real
// process-exit/static-destruction teardown; a cache recreated after a
// completed module shutdown, in the SAME live process, is not a scenario
// production code creates -- see the two reactivation hooks above, both
// of which explicitly re-arm the flag first). Measured: even
// explicit_global_cache_shutdown_is_clean() alone, several cases earlier
// in main()'s default run, is enough to poison every case after it this
// way -- not just the two shutdown-poisoning cases poisoning each other.

// llama.cpp-me60: shared setup for both shutdown-repro cases below --
// allocate one oneDNN Graph-scratch DIRECT buffer of `size` bytes on `dq`,
// free it with a short device-kernel release event (so
// onednn_graph_scratch_free() takes the arming branch and actually
// allocates the completion-flag slab both cases target), then wait for
// both the release event and the separate, asynchronous marker kernel that
// arms its flag slot (see onednn_graph_scratch_pool_entry::flag_slot's own
// comment in unified-cache.hpp) so this parks a genuine, complete registry
// row rather than one still mid-arming when a caller proceeds. Every step
// is already TEST_ASSERT'd here, so a caller only needs to
// TEST_ASSERT(park_one_direct_entry(...), ...) once around the whole call
// -- the printed failure message names which specific step failed.
static bool park_one_direct_entry(unified_cache * cache, sycl::queue & dq, size_t size) {
    void * ptr = cache->onednn_graph_scratch_alloc(size, 256, &dq);
    TEST_ASSERT(ptr != nullptr, "DIRECT allocation for the shutdown-repro fixture failed (setup, not the regression)");

    int * cell = sycl::malloc_device<int>(1, dq);
    TEST_ASSERT(cell != nullptr, "device marker cell allocation failed (setup, not the regression)");
    dq.memset(cell, 0, sizeof(int)).wait();
    sycl::event release_event = submit_spin_kernel(dq, cell, 2000000);

    cache->onednn_graph_scratch_free(ptr, &release_event);

    release_event.wait();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2500);
    while (!cache->onednn_graph_scratch_pool_entry_flag_true_for_test(size) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    TEST_ASSERT(cache->onednn_graph_scratch_pool_entry_flag_true_for_test(size),
                "marker flag did not arm within 2500 ms -- setup, not the regression");
    sycl::free(cell, dq);
    return true;
}

static bool onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim(sycl::queue & q) {
    TEST_BEGIN("onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim");

    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    TEST_ASSERT(cache->onednn_graph_scratch_direct_outstanding_bytes() == 0,
                "cache did not start from a clean slate -- a prior case left DIRECT Graph-scratch bytes "
                "outstanding, which would make this case's own isolation step meaningless");
    TEST_ASSERT(!ggml_sycl_is_shutting_down(),
                "g_sycl_shutting_down is already set -- this process is not the fresh one this case requires; "
                "its own shutdown_resources() would take the abandon branch this case does not target");

    sycl::queue & dq = cache->get_queue();

    // A different size than the other case's 300 MiB: not load-bearing --
    // the REAL protection is that each case runs in its own ctest-launched
    // process (see the comment above this function), so nothing else in
    // this binary can have parked a DIRECT entry or left
    // g_sycl_shutting_down set before this case runs. The two asserts just
    // above are a belt-and-suspenders check that would catch it loudly if
    // that process isolation were ever broken (e.g. by a future edit
    // calling this case from main()'s default path again); the size
    // difference from the other case's 300 MiB just keeps the two cases'
    // failures distinguishable by size in a log if ever needed. Same
    // zone-floor/cap reasoning as the other case's own comment.
    constexpr size_t size = 320ull * 1024 * 1024;

    TEST_ASSERT(park_one_direct_entry(cache, dq, size),
                "parking one complete DIRECT Graph-scratch entry failed (setup, not the regression)");

    // THE ISOLATION STEP: the parked entry is already complete (waited on
    // above), so onednn_graph_scratch_clear_pool_locked() takes its
    // immediate real-release branch, destructing the buffer's own
    // EXTERNAL_EXACT allocation control synchronously -- satisfying
    // snapshot_allocation_controls("pre-cache-teardown", ...,
    // preteardown=true)'s admissibility check regardless of the
    // pool-census defect the other case in this file demonstrates.
    // device=0: ONEAPI_DEVICE_SELECTOR remaps the selected B50/B70 to
    // runtime device 0, same convention every other case in this file
    // already uses.
    unified_cache_reclaim_onednn_graph_scratch_pool(0, "test isolation (llama.cpp-me60)");

    // THE ACTUAL REGRESSION CHECK for this case.
    TEST_ASSERT(shutdown_unified_cache(),
                "module shutdown refused even after the DIRECT pool was explicitly reclaimed ahead of the "
                "pre-teardown census -- the flag slab's OWN registry row still fails the later sweep "
                "(llama.cpp-me60 F1); look for \"[UNIFIED-CACHE] runtime allocation registry still populated "
                "at shutdown boundary\" in stderr above");

    TEST_PASS();
    return true;
}

// llama.cpp-me60 F1, STEP 0: the oneDNN Graph-scratch completion-flag
// slab (onednn_graph_scratch_flag_slab_owner_, allocated lazily by
// onednn_graph_scratch_ensure_flag_slab_locked() the first time a DIRECT
// entry is parked with a real release event) is adopted into the global
// runtime-allocation registry as a bootstrap CACHE_BACKING control. Before
// F1's fix, shutdown_resources() never released it on the normal path --
// it was left to ~unified_cache() member destruction, which ran AFTER
// unified_cache_shutdown_retryable_postconditions_clean()'s sweep. That
// sweep (see runtime_allocation_owned_by_cache_snapshot()) accepts only
// rows classified via contains_pinned()/contains_pinned_backing_allocation()
// against host_arena_ -- it never consults allocation_control_class -- so
// the slab's row, host USM allocated OUTSIDE the pinned pool, failed it
// regardless of being CACHE_BACKING. shutdown_unified_cache() then
// returned false with "[UNIFIED-CACHE] runtime allocation registry still
// populated at shutdown boundary" once any DIRECT entry had ever been
// parked with a real event in this process.
//
// This case targets exactly that sweep: park one DIRECT Graph-scratch
// entry with a device-kernel release event (so the slab is allocated and
// armed, mirroring a real oneDNN free callback -- a nullptr event takes
// the "complete unconditionally" branch and never allocates the slab at
// all, per onednn_graph_scratch_free()'s own comment), then call the same
// module-level shutdown_unified_cache() every other case in this file
// already asserts succeeds.
//
// EXPECTED ON UNFIXED CODE (pre-llama.cpp-me60): FAILS with the registry
// message above. F1's fix releases the slab under
// onednn_graph_scratch_mutex_ inside shutdown_resources()'s normal path,
// so this case passes -- but ONLY when it is the first thing in the
// process to ever call shutdown_unified_cache() (see below); it is not a
// general property of a fixed build.
//
// RUNS ONLY VIA ITS OWN ctest REGISTRATION
// (sycl-runtime-alloc-flag-slab-released-at-shutdown, `--case <name>`), in
// a fresh process, with nothing run before it -- NOT from main()'s default
// (no-argument) run, and not merely placed after the sibling case either
// (an earlier version of this comment said "run last" and main() called it
// that way). Two hardware findings established why, matching the sibling
// case's own comment (onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim(),
// above in this file) in full -- reproduced here rather than cross-referenced,
// since this case is the one whose own GGML_ABORT that finding names:
//
// (1) This case's own shutdown_unified_cache() call below is EXPECTED to
// be refused on unfixed code (that is the whole point of this case), and a
// refused call is not retryable-safe here -- it leaves the cache PARTIALLY
// torn down: its own parked buffer's control keeps the cache registered in
// the per-device cache map, but a subsequent onednn_graph_scratch_alloc() on that
// same cache resolves off-device and hits GGML_ABORT at
// onednn_graph_scratch_alloc_direct_locked(). So nothing may run in this
// process after this case, on unfixed code.
//
// (2) Independent of (1), and observed even on FIXED code: any earlier
// case in this process that completes a successful shutdown_unified_cache()
// call leaves g_sycl_shutting_down == true (its own store(true) at that
// call's very end), reset to false only mid-way through the NEXT such
// call -- after ITS OWN shutdown_resources() loop has already run using
// the stale true value. A cache this case's own get_unified_cache(q) call
// lazily recreates after such a prior shutdown does not reset that flag
// (only the production reactivation hooks --
// prepare_unified_cache_for_module_use(), called from
// ggml_backend_sycl_commit_reactivate() and ggml_backend_sycl_reg() -- do
// that; this file bypasses that whole protocol). So this case's own
// shutdown_resources() call takes the "SYCL might already be gone" ABANDON
// branch instead of the normal release path, leaving the flag slab's
// registry row behind for a reason that has nothing to do with F1 --
// verified on hardware (build-me60-4/5): even
// explicit_global_cache_shutdown_is_clean() alone, several cases earlier
// in main()'s default run, was enough to make this case fail this way on
// an otherwise-fixed build.
static bool onednn_graph_scratch_flag_slab_released_at_module_shutdown(sycl::queue & q) {
    TEST_BEGIN("onednn_graph_scratch_flag_slab_released_at_module_shutdown");

    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    TEST_ASSERT(cache->onednn_graph_scratch_direct_outstanding_bytes() == 0,
                "cache did not start from a clean slate -- a prior case left DIRECT Graph-scratch bytes "
                "outstanding, which would make this case's own park-and-shutdown check meaningless");
    TEST_ASSERT(!ggml_sycl_is_shutting_down(),
                "g_sycl_shutting_down is already set -- this process is not the fresh one this case requires; "
                "its own shutdown_resources() would take the abandon branch this case does not target");

    sycl::queue & dq = cache->get_queue();

    // Must exceed the natural ~256 MB no-model ONEDNN zone floor (measured
    // on hardware -- see test-sycl-onednn-graph-scratch-direct.cpp's own
    // kZoneFloorBytes comment: GGML_SYCL_ONEDNN_GRAPH_ZONE_MB does not
    // shrink it for a no-model arena) so this request deterministically
    // takes the DIRECT (non-arena) path that can allocate/arm the flag
    // slab, and stays under the 700 MB cap main() raises via
    // GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB before first cache use.
    constexpr size_t size = 300ull * 1024 * 1024;

    TEST_ASSERT(park_one_direct_entry(cache, dq, size),
                "parking one complete DIRECT Graph-scratch entry failed (setup, not the regression)");

    // THE ACTUAL REGRESSION CHECK.
    TEST_ASSERT(shutdown_unified_cache(),
                "module shutdown refused once a DIRECT Graph-scratch entry had been parked -- the "
                "completion-flag slab's registry row is not released inside shutdown_resources() "
                "(llama.cpp-me60 F1); look for \"[UNIFIED-CACHE] runtime allocation registry still populated at "
                "shutdown boundary\" in stderr above");

    TEST_PASS();
    return true;
}
#endif  // GGML_SYCL_DNNL

static bool independent_exact_token_defers_owned_release(sycl::queue & q) {
    TEST_BEGIN("B50_B70_independent_exact_token_defers_owned_release");
    unified_cache * cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    if (!cache->arena_active()) {
        TEST_PASS();
        return true;
    }

    alloc_request req{};
    req.queue                                      = &q;
    req.size                                       = 4096;
    req.intent.role                                = alloc_role::GRAPH_TMP;
    req.intent.category                            = runtime_category::GRAPH;
    req.intent.constraints.must_device             = true;
    req.intent.constraints.prefer_vram_zone         = vram_zone_id::RUNTIME;

    alloc_handle allocation{};
    TEST_ASSERT(unified_alloc(req, &allocation), "exact arena allocation failed");
    if (!allocation.zone_managed || allocation.vram_zone == vram_zone_id::COUNT) {
        TEST_ASSERT(unified_free(allocation), "non-arena fixture cleanup failed");
        TEST_PASS();
        return true;
    }

    // Snapshot the copyable identity before consuming the legacy ownership
    // token. Promotion deliberately rejects lvalues so there cannot be two
    // apparent owners of the same registry row.
    const alloc_metadata exact = allocation.metadata();
    void * const ptr           = exact.ptr;
    const size_t used_before   = cache->zone_used(exact.zone);
    mem_handle owner           = detail::from_legacy_owned_alloc(std::move(allocation));
    TEST_ASSERT(owner.valid(), "owning arena handle construction failed");
    mem_handle independent = mem_handle::from_arena_zone(
        static_cast<int>(exact.zone), exact.offset, exact.size, exact.device,
        exact.generation, exact.alloc_id, exact.extent,
        cache->arena_authority_snapshot(exact.zone));
    TEST_ASSERT(independent.valid() && independent.resolve().ptr == ptr,
                "independent exact token admission failed");

    auto coordinator = unified_allocation_release_coordinator(exact.device);
    TEST_ASSERT(coordinator != nullptr, "owning arena handle omitted its release coordinator");
    owner = {};
    alloc_metadata looked{};
    TEST_ASSERT(unified_lookup(ptr, &looked), "refused owner release erased the runtime record");
    TEST_ASSERT(coordinator->retry_count() == 1, "refused owner release lost durable retry ownership");
    TEST_ASSERT(cache->zone_used(exact.zone) == used_before,
                "refused owner release returned the TLSF block");

    TEST_ASSERT(coordinator->process_retries() == 0,
                "retry freed allocation while independent token remained live");
    TEST_ASSERT(unified_lookup(ptr, &looked), "retry erased the record while independent token remained live");
    independent = {};
    TEST_ASSERT(coordinator->process_retries() == 1, "final-token retry did not release the allocation");
    TEST_ASSERT(!unified_lookup(ptr, &looked), "final-token sync did not erase the runtime record");
    TEST_ASSERT(cache->zone_used(exact.zone) < used_before,
                "final-token sync did not reclaim the TLSF block");
    TEST_PASS();
    return true;
}

// A cache whose arena is reserved by hand stores no ONEDNN zone plan (only ensure_planned_arena_zones() stores one), so
// its zone is bounded by its capacity alone and a pair that fits the zone is served. The device's own cache, built by
// the earlier cases, holds the plan its ensure stored, whose Graph floor can exceed this 8 MiB zone; judged by that
// plan, as a plan stored per device judged it, the zone's pair bound came out 0 and every pair was refused. A note
// says whether that plan was there to be misread, and so whether the run shows anything about where the plan lives.
static bool hand_built_arena_is_judged_by_its_own_zone_plan(sycl::queue & q) {
    TEST_BEGIN("hand_built_arena_is_judged_by_its_own_zone_plan");
    constexpr size_t mib          = 1024ull * 1024ull;
    const size_t     max_alloc    = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache *  device_cache = get_unified_cache(q);
    TEST_ASSERT(device_cache != nullptr, "no cache for the test device");
    const zone_onednn_plan device_plan = device_cache->onednn_zone_plan_load();

    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    if (cache.arena_active()) {
        // The constructor's own ensure built the zones (a small GGML_SYCL_COMPUTE_ARENA_MB), so they carry its plan and
        // none can be reserved by hand over them.
        fprintf(stderr, "(the constructor built the arena; no hand-built zone) ");
        TEST_PASS();
        return true;
    }
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        fprintf(stderr, "(arena reservation failed; no hand-built zone) ");
        TEST_PASS();
        return true;
    }
    const size_t zone = cache.zone_capacity(vram_zone_id::ONEDNN);
    TEST_ASSERT(zone_onednn_plan_same(cache.onednn_zone_plan_load(), zone_onednn_plan()),
                "a hand-built arena holds a zone plan nothing stored for it");
    if (zone_onednn_pp_pair_bound(zone, device_plan.bare_bytes, device_plan.graph_floor_bytes) < 2ull * mib) {
        fprintf(stderr, "(the device cache's plan, Graph floor %.1f MB, would bound this %.1f MB zone below the pair) ",
                device_plan.graph_floor_bytes / (1024.0 * 1024.0), zone / (1024.0 * 1024.0));
    } else {
        fprintf(stderr,
                "(the device cache's plan leaves this zone room for the pair; this run shows nothing about where "
                "the plan lives) ");
    }
    TEST_ASSERT(cache.reserve_onednn_scratch(1ull * mib, 1ull * mib), "a pair fitting the hand-built zone was refused");
    TEST_ASSERT(cache.shutdown_resources(), "arena shutdown did not release every self-owned allocation");
    TEST_PASS();
    return true;
}

static bool arena_owned_shutdown_and_lifecycle_serialization(sycl::queue & q) {
    TEST_BEGIN("arena_owned_shutdown_and_lifecycle_serialization");
    constexpr size_t mib = 1024ull * 1024ull;
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_PASS();
        return true;
    }

    // Two settlers may complete in either order, but neither may publish an
    // older generation over the other. A subsequent allocation proves OPEN was
    // published only after the complete reset transaction.
    std::thread a([&] { cache.test_zone_boundary_check(vram_zone_id::ONEDNN); });
    std::thread b([&] { cache.test_zone_boundary_check(vram_zone_id::ONEDNN); });
    a.join();
    b.join();

    TEST_ASSERT(cache.reserve_scratch_pool(1ull * mib), "real arena scratch pool reserve failed");
    TEST_ASSERT(cache.reserve_onednn_scratch(1ull * mib, 1ull * mib), "real arena oneDNN reserve failed");
    TEST_ASSERT(cache.reserve_persistent_scratch("shutdown-owner", 1ull * mib),
                "real arena persistent reserve failed");
    TEST_ASSERT(cache.shutdown_resources(), "arena shutdown did not release every self-owned allocation");
    TEST_ASSERT(!cache.arena_active(), "arena remained active after exact owner teardown");
    TEST_PASS();
    return true;
}

static bool arena_shutdown_drains_dma_and_bcs(sycl::queue & q) {
    TEST_BEGIN("B50_shutdown_drains_dma_and_bcs");
    constexpr size_t mib = 1024ull * 1024ull;
    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    std::atomic<unsigned> completed{ 0 };
    auto submit_marker = [&](sycl::queue & exact_queue) {
        exact_queue.submit([&](sycl::handler & cgh) {
            cgh.host_task([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                completed.fetch_add(1, std::memory_order_release);
            });
        });
    };
    submit_marker(cache.get_dma_queue());
    submit_marker(cache.get_bcs_queue());
    TEST_ASSERT(cache.shutdown_resources(), "shutdown failed while exact transfer queues were in flight");
    TEST_ASSERT(completed.load(std::memory_order_acquire) == 2,
                "shutdown released owners before DMA/BCS terminal work completed");
    TEST_PASS();
    return true;
}

static bool exact_scratch_shutdown_record_survives_for_retry(sycl::queue & q) {
    TEST_BEGIN("B50_exact_scratch_shutdown_record_survives_for_retry");
    constexpr size_t mib = 1024ull * 1024ull;
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_PASS();
        return true;
    }
    TEST_ASSERT(cache.reserve_scratch_pool(1ull * mib), "scratch owner reserve failed");
    mem_handle retained = cache.test_scratch_pool_owner();
    TEST_ASSERT(retained.resolve(), "copied arena owner did not acquire its authority lease");
    const mem_handle_debug_info owner = retained.debug_info();
    TEST_ASSERT(retained.is_arena(), "scratch fixture returned a non-arena direct owner");
    TEST_ASSERT(owner.canonical_allocation_id != 0, "scratch fixture lost its exact allocation id");
    TEST_ASSERT(owner.generation != 0 && owner.canonical_generation == owner.generation,
                "scratch fixture lost its exact arena generation");
    TEST_ASSERT(owner.zone_id == static_cast<int>(vram_zone_id::WEIGHT),
                "scratch fixture returned the wrong arena zone");
    TEST_ASSERT(owner.canonical_extent == 2ull * mib && owner.size == owner.canonical_extent,
                "scratch fixture returned the wrong exact allocation extent");

    const size_t exact_block_used = cache.zone_used(vram_zone_id::WEIGHT);
    unified_cache_test_set_arena_drain_timeout_ms(10);
    TEST_ASSERT(!cache.shutdown_resources(), "shutdown unexpectedly freed an exact allocation with a retained lease");
    TEST_ASSERT(cache.arena_active() && cache.chunk_count() > 0,
                "refused shutdown discarded physical chunks needed by retry");
    TEST_ASSERT(cache.scratch_pool_capacity() == 1ull * mib,
                "refused shutdown discarded scratch allocation geometry");
    TEST_ASSERT(cache.zone_used(vram_zone_id::WEIGHT) == exact_block_used,
                "refused shutdown returned the exact TLSF block to the allocator");
    TEST_ASSERT(retained.resolve(), "refused shutdown removed the exact authority record");
    retained = {};
    unified_cache_test_set_arena_drain_timeout_ms(5000);
    TEST_ASSERT(cache.shutdown_resources(), "destroy retry did not drain the persisted authority");
    TEST_ASSERT(!cache.arena_active(), "successful retry left physical arena chunks live");
    TEST_PASS();
    return true;
}

static bool scratch_regrow_refusal_preserves_exact_record(sycl::queue & q) {
    TEST_BEGIN("B70_scratch_regrow_refusal_preserves_exact_record");
    constexpr size_t mib = 1024ull * 1024ull;
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_PASS();
        return true;
    }
    TEST_ASSERT(cache.reserve_scratch_pool(1ull * mib), "initial scratch reserve failed");
    mem_handle external = cache.test_scratch_pool_owner();
    resolved_ptr old = external.resolve();
    TEST_ASSERT(old && old.extent == 2ull * mib, "external scratch owner did not resolve exact allocation");
    const size_t exact_block_used = cache.zone_used(vram_zone_id::WEIGHT);

    TEST_ASSERT(!cache.reserve_scratch_pool(2ull * mib),
                "scratch regrow replaced an allocation with an external exact lease");
    TEST_ASSERT(cache.scratch_pool_capacity() == 1ull * mib,
                "refused regrow published new scratch geometry");
    TEST_ASSERT(cache.zone_used(vram_zone_id::WEIGHT) == exact_block_used,
                "refused regrow released or duplicated the TLSF block");
    TEST_ASSERT(external.resolve().ptr == old.ptr,
                "refused regrow removed the exact record or changed its pointer");
    void * still_usable = cache.get_scratch(256);
    TEST_ASSERT(still_usable != nullptr, "old scratch allocation became unusable after refused regrow");
    cache.return_scratch(still_usable, 256);

    external = {};
    TEST_ASSERT(cache.reserve_scratch_pool(2ull * mib), "scratch regrow retry failed after external lease release");
    TEST_ASSERT(cache.scratch_pool_capacity() == 2ull * mib,
                "successful regrow did not publish the requested geometry");
    TEST_ASSERT(cache.shutdown_resources(), "scratch regrow fixture did not shut down cleanly");
    TEST_PASS();
    return true;
}

static bool concurrent_settle_destroy_closing_wins(sycl::queue & q) {
    TEST_BEGIN("B70_concurrent_settle_destroy_closing_wins");
    constexpr size_t mib = 1024ull * 1024ull;
    const size_t max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache cache(q, 64ull * mib, 0, 0, 0);
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_PASS();
        return true;
    }

    unified_cache_test_pause_zone_settle(true);
    std::thread settler([&] { cache.test_zone_boundary_check(vram_zone_id::ONEDNN); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!unified_cache_test_zone_settle_reached() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (!unified_cache_test_zone_settle_reached()) {
        unified_cache_test_pause_zone_settle(false);
        settler.join();
        TEST_FAIL("settle did not reach the deterministic RESETTING barrier");
    }

    bool destroyed = false;
    std::thread destroyer([&] { destroyed = cache.shutdown_resources(); });
    while (!unified_cache_test_arena_destroy_closing_reached() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool closing_won = unified_cache_test_arena_destroy_closing_reached();
    unified_cache_test_pause_zone_settle(false);
    settler.join();
    destroyer.join();
    TEST_ASSERT(closing_won, "destroy did not take ownership from the concurrent RESETTING transaction");
    TEST_ASSERT(destroyed && !cache.arena_active(), "settler reopened the arena after CLOSING won");
    TEST_PASS();
    return true;
}

// An arm that cannot build its fixture on this device measures nothing: it says why and marks the run skipped, and the
// case then exits 77, never 0.
#define TEST_SKIP(skipped, why)                \
    do {                                       \
        fprintf(stderr, "SKIPPED: %s\n", why); \
        (skipped) = true;                      \
        return true;                           \
    } while (0)

// Every zone's live authority and the generation it was minted at. A handle minted in a zone resolves only through
// that authority object at that generation, so an arena whose incarnations are all unchanged resolves every handle it
// resolved before to the same pointer.
struct arena_incarnations {
    std::array<std::shared_ptr<arena_authority>, static_cast<size_t>(vram_zone_id::COUNT)> authority{};
    std::array<uint64_t, static_cast<size_t>(vram_zone_id::COUNT)>                         generation{};
};

static arena_incarnations arena_incarnations_of(const unified_cache & cache) {
    arena_incarnations seen;
    for (size_t z = 0; z < static_cast<size_t>(vram_zone_id::COUNT); ++z) {
        seen.authority[z]  = cache.arena_authority_snapshot(static_cast<vram_zone_id>(z));
        seen.generation[z] = cache.arena_generation(static_cast<vram_zone_id>(z));
    }
    return seen;
}

// True when every zone has the authority object and generation it had, and that authority admits again.
static bool arena_incarnations_unchanged(const unified_cache & cache, const arena_incarnations & before) {
    const arena_incarnations now = arena_incarnations_of(cache);
    for (size_t z = 0; z < static_cast<size_t>(vram_zone_id::COUNT); ++z) {
        if (now.authority[z] != before.authority[z] || now.generation[z] != before.generation[z]) {
            return false;
        }
        if (now.authority[z] && !now.authority[z]->accepts(now.generation[z])) {
            return false;
        }
    }
    return true;
}

// The SCRATCH zone of the arena serves a new allocation, inside the arena, and takes it back.
static bool arena_serves_a_new_allocation(unified_cache & cache) {
    constexpr size_t mib = 1024ull * 1024ull;
    if (!cache.reserve_persistent_scratch("arena-teardown-restored", 1ull * mib)) {
        return false;
    }
    void * const fresh  = cache.get_persistent_scratch("arena-teardown-restored");
    const bool   inside = fresh != nullptr && cache.vram_owns(fresh);
    cache.release_persistent_scratch("arena-teardown-restored");
    return inside;
}

// A teardown refused because allocations are live in the arena leaves the arena as it was. The fixture holds the
// scratch pool's exact allocation through a copy of its owner and asks the arena to tear itself down; the refusal is
// the positive control. Then the arena is as it was: the held handle resolves to the pointer it resolved to before,
// every zone keeps its authority and generation, and the SCRATCH zone serves a new allocation inside the arena.
// Mutant: the teardown closing the allocator groups and retiring the authorities before it finds the allocations, and
// not putting them back (the held handle stops resolving, the SCRATCH zone refuses the allocation).
static bool arena_teardown_live_allocations_restores(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_live_allocations_restores");
    constexpr size_t mib       = 1024ull * 1024ull;
    const size_t     max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache    cache(q, 64ull * mib, 0, 0, 0);
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no teardown was measured");
    }
    TEST_ASSERT(cache.reserve_scratch_pool(1ull * mib), "the fixture's scratch pool was not reserved");
    mem_handle         held   = cache.test_scratch_pool_owner();
    const resolved_ptr before = held.resolve();
    TEST_ASSERT(before, "the held scratch pool owner did not resolve before the teardown");
    const arena_incarnations incarnations = arena_incarnations_of(cache);
    const int                chunks       = cache.chunk_count();

    TEST_ASSERT(!cache.arena_destroy(), "the teardown freed an arena a live allocation is in");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::LIVE_ALLOCATIONS,
                "the teardown did not record the live allocation as its cause");
    TEST_ASSERT(cache.arena_active() && cache.chunk_count() == chunks, "the refused teardown changed the chunks");
    const resolved_ptr after = held.resolve();
    TEST_ASSERT(after && after.ptr == before.ptr,
                "the held handle does not resolve to the pointer it resolved to before the refused teardown");
    TEST_ASSERT(arena_incarnations_unchanged(cache, incarnations),
                "a zone's authority or generation is not the one it had before the refused teardown");
    TEST_ASSERT(arena_serves_a_new_allocation(cache), "the arena refused a new allocation after the refused teardown");

    held = {};
    TEST_ASSERT(cache.shutdown_resources(), "the arena was not torn down once its last holder was released");
    TEST_ASSERT(!cache.arena_active(), "the completed teardown left the arena active");
    TEST_PASS();
    return true;
}

// Declared after an arm's cache, so it runs before the cache's destructor. An arm that returns early, or whose
// teardown never reached a failure hook it armed, leaves that hook armed: the hooks are disarmed first, so none fires
// here, in a later arm or in the destructor. Then the arena is torn down, a second time if the first is refused, so
// the destructor finds no arena (a shutdown it cannot complete aborts the process).
struct arena_teardown_retry {
    unified_cache & cache;

    ~arena_teardown_retry() {
        unified_cache_test_disarm_arena_teardown_failures();
        if (!cache.shutdown_resources()) {
            (void) cache.shutdown_resources();
        }
    }
};

// GGML_SYCL_COMPUTE_ARENA_MB, put back as it was when the arm returns.
struct compute_arena_mb_restore {
    bool        had;
    std::string value;

    compute_arena_mb_restore() {
        const char * saved = std::getenv("GGML_SYCL_COMPUTE_ARENA_MB");
        had                = saved != nullptr;
        value              = saved != nullptr ? saved : "";
    }

    ~compute_arena_mb_restore() { set_env_var("GGML_SYCL_COMPUTE_ARENA_MB", had ? value.c_str() : nullptr); }
};

// The teardown's waits for leases, on its retired authorities and on each chunk, shortened to 100 ms for an arm that
// holds a chunk lease through a teardown, so the arm does not wait out the 5000 ms default; put back when it returns.
struct arena_lease_wait_short {
    arena_lease_wait_short() { unified_cache_test_set_arena_drain_timeout_ms(100); }

    ~arena_lease_wait_short() { unified_cache_test_set_arena_drain_timeout_ms(5000); }
};

// A chunk lease an arm holds through the cache's chunk lease API (the primitive mem_handle::from_chunk_ptr takes,
// called directly because that bridge finds the per-device cache, not the arm's own). Declared after the arm's
// arena_teardown_retry, so the lease is released before the retry's teardown on every return.
struct held_chunk_lease {
    unified_cache & cache;
    int             chunk = -1;

    ~held_chunk_lease() { release(); }

    void release() {
        if (chunk >= 0) {
            cache.arena_release_chunk_lease(chunk);
            chunk = -1;
        }
    }
};

// One pointer inside each chunk of the cache's arena, by chunk index: the first byte of the logical range each chunk
// starts, or nullptr for a chunk the scan finds no pointer in.
static std::vector<const void *> arena_pointer_in_each_chunk(unified_cache & cache) {
    constexpr size_t          mib = 1024ull * 1024ull;
    std::vector<const void *> in_chunk(static_cast<size_t>(cache.chunk_count()), nullptr);
    for (size_t offset = 0; offset < cache.arena_total_size(); offset += 1ull * mib) {
        const void * p     = cache.offset_to_ptr(offset);
        const int    chunk = cache.arena_find_chunk(p);
        if (chunk >= 0 && in_chunk[static_cast<size_t>(chunk)] == nullptr) {
            in_chunk[static_cast<size_t>(chunk)] = p;
        }
    }
    return in_chunk;
}

// A rebuild whose teardown the device refuses at its first free leaves the arena as it was, and a later load tries the
// rebuild again. The arena's 8 MiB zones are smaller than the 32 MiB SCRATCH zone the ensure asks for, and nothing
// holds it, so the ensure tears it down to rebuild it; the device refuses to free the first chunk (the PRIVATE_TESTING
// free failure, armed for a teardown that has freed nothing). The ensure's refusal naming that cause is the positive
// control. Then the arena is as it was: the same chunks and size, every zone's authority and generation, and the
// SCRATCH zone serves a new allocation inside it. A second ensure tries the teardown again, and that teardown
// completes.
// Engagement: the arena is reserved with zones smaller than the ensure's, so the ensure must rebuild it.
// Mutants: the refused first free not restoring (the authorities stay retired, so the incarnation check finds them
// changed and the SCRATCH zone refuses the new allocation; FIRST_FREE_FAILED is not read as closed, so the second
// ensure would still try the teardown and complete it); the cause read as closed (the second ensure refuses without
// trying).
static bool arena_teardown_first_free_refusal_restores(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_first_free_refusal_restores");
    constexpr size_t         mib       = 1024ull * 1024ull;
    const size_t             max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    compute_arena_mb_restore env;
    unified_cache            cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry     retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no teardown was measured");
    }
    set_env_var("GGML_SYCL_COMPUTE_ARENA_MB", "32");
    TEST_ASSERT(cache.zone_capacity(vram_zone_id::SCRATCH) < 32ull * mib,
                "the arena's SCRATCH zone already holds the ensure's 32 MiB, so no rebuild is needed");
    const int                chunks_before = cache.chunk_count();
    const size_t             size_before   = cache.arena_total_size();
    const arena_incarnations incarnations  = arena_incarnations_of(cache);

    unified_cache_test_fail_next_arena_free(0);
    zone_arena_refusal first;
    TEST_ASSERT(!cache.ensure_planned_arena_zones(&first), "the ensure succeeded over a teardown that refused");
    TEST_ASSERT(first.reason == zone_arena_refusal_reason::TEARDOWN_REFUSED &&
                    first.teardown == zone_arena_teardown_refusal::FIRST_FREE_FAILED,
                "the ensure did not refuse for the first free the device refused");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::FIRST_FREE_FAILED,
                "the cache does not record why its teardown refused");
    TEST_ASSERT(cache.arena_active() && cache.chunk_count() == chunks_before && cache.arena_total_size() == size_before,
                "the refused teardown changed the arena's chunks or size, or an arena was reserved over it");
    TEST_ASSERT(arena_incarnations_unchanged(cache, incarnations),
                "a zone's authority or generation is not the one it had before the refused teardown");
    TEST_ASSERT(arena_serves_a_new_allocation(cache), "the arena refused a new allocation after the refused teardown");

    // The retry's reserve may still refuse for this device's budget; what it must not do is refuse for the teardown.
    zone_arena_refusal second;
    const bool         rebuilt = cache.ensure_planned_arena_zones(&second);
    TEST_ASSERT(rebuilt || second.reason != zone_arena_refusal_reason::TEARDOWN_REFUSED,
                "a second ensure refused for the teardown instead of trying it again");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::NONE,
                "the second ensure's teardown did not complete");
    TEST_PASS();
    return true;
}

// A teardown whose wait for the leases on its retired authorities times out leaves the arena as it was. Nothing in the
// fixture holds a lease, so the PRIVATE_TESTING drain failure makes the wait read as timed out; the refusal naming
// that cause is the positive control. Then the arena is as it was, and a second teardown completes.
// Mutant: the drain timeout returning without restore() (the authorities stay retired and the groups CLOSING, so the
// SCRATCH zone refuses the allocation).
static bool arena_teardown_drain_timeout_restores(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_drain_timeout_restores");
    constexpr size_t     mib       = 1024ull * 1024ull;
    const size_t         max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache        cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no teardown was measured");
    }
    const int                chunks       = cache.chunk_count();
    const arena_incarnations incarnations = arena_incarnations_of(cache);

    unified_cache_test_fail_next_arena_drain();
    TEST_ASSERT(!cache.arena_destroy(), "the teardown freed the arena although its drain timed out");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::DRAIN_TIMEOUT,
                "the teardown did not record the timed-out drain as its cause");
    TEST_ASSERT(cache.arena_active() && cache.chunk_count() == chunks, "the refused teardown changed the chunks");
    TEST_ASSERT(arena_incarnations_unchanged(cache, incarnations),
                "a zone's authority or generation is not the one it had before the refused teardown");
    TEST_ASSERT(arena_serves_a_new_allocation(cache), "the arena refused a new allocation after the refused teardown");

    TEST_ASSERT(cache.arena_destroy(), "a second teardown did not complete");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::NONE && !cache.arena_active(),
                "the completed teardown left the arena active or the earlier cause recorded");
    TEST_PASS();
    return true;
}

// A refused teardown that interrupted a settle finishes that settle. The settle of the ONEDNN group is held at its
// RESETTING barrier, with ONEDNN's authority retired (the positive control that it is in flight); then a teardown runs
// and its drain is made to time out. The teardown's restore finds the settle's drains empty and finishes the settle
// itself, before the settle goes on: ONEDNN has a new authority that admits at the zone's generation, and every other
// zone keeps its authority and generation. The released settle sees the group OPEN and publishes nothing over it.
// Mutant: the restore giving the group back RESETTING without finishing the settle (ONEDNN has no authority while the
// settle is held).
static bool arena_teardown_restore_finishes_an_interrupted_settle(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_restore_finishes_an_interrupted_settle");
    constexpr size_t     mib       = 1024ull * 1024ull;
    const size_t         max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache        cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no teardown was measured");
    }
    const std::shared_ptr<arena_authority> settled = cache.arena_authority_snapshot(vram_zone_id::ONEDNN);

    // Declared after the retry, so it runs first: a held settle is let go and joined on every return.
    struct settle_release {
        std::thread settler;

        ~settle_release() {
            unified_cache_test_pause_zone_settle(false);
            if (settler.joinable()) {
                settler.join();
            }
        }
    } held;

    unified_cache_test_pause_zone_settle(true);
    held.settler        = std::thread([&] { cache.test_zone_boundary_check(vram_zone_id::ONEDNN); });
    // A deadline, not a measure: the settle thread reaches its barrier at once on an idle host, and this host's
    // permanent load must not turn a slow start into a failure.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!unified_cache_test_zone_settle_reached() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    TEST_ASSERT(unified_cache_test_zone_settle_reached(), "the settle did not reach its RESETTING barrier");
    TEST_ASSERT(settled && cache.arena_authority_snapshot(vram_zone_id::ONEDNN) == nullptr,
                "the held settle did not retire ONEDNN's authority");
    const arena_incarnations others = arena_incarnations_of(cache);

    unified_cache_test_fail_next_arena_drain();
    TEST_ASSERT(!cache.arena_destroy(), "the teardown freed the arena although its drain timed out");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::DRAIN_TIMEOUT,
                "the teardown did not record the timed-out drain as its cause");
    const std::shared_ptr<arena_authority> finished = cache.arena_authority_snapshot(vram_zone_id::ONEDNN);
    TEST_ASSERT(finished && finished != settled && finished->accepts(cache.arena_generation(vram_zone_id::ONEDNN)),
                "the refused teardown did not finish the settle it interrupted");
    const arena_incarnations now = arena_incarnations_of(cache);
    for (size_t z = 0; z < static_cast<size_t>(vram_zone_id::COUNT); ++z) {
        if (z == static_cast<size_t>(vram_zone_id::ONEDNN)) {
            continue;
        }
        TEST_ASSERT(now.authority[z] == others.authority[z] && now.generation[z] == others.generation[z] &&
                        (!now.authority[z] || now.authority[z]->accepts(now.generation[z])),
                    "a zone the settle did not touch is not as it was before the refused teardown");
    }

    unified_cache_test_pause_zone_settle(false);
    held.settler.join();
    TEST_ASSERT(cache.arena_authority_snapshot(vram_zone_id::ONEDNN) == finished,
                "the released settle published over the authority the refused teardown published");
    TEST_ASSERT(arena_serves_a_new_allocation(cache), "the arena refused a new allocation after the refused teardown");
    TEST_PASS();
    return true;
}

// A teardown that cannot free a chunk after it freed one leaves the arena partly freed and closed. The arena is
// reserved in three chunks (a 48 MiB per-chunk cap under a 128 MiB budget), and the device refuses to free the second
// chunk, after the first was freed (the PRIVATE_TESTING free failure, armed for a teardown that has freed one chunk).
// The refusal naming FREE_FAILED is the positive control. The teardown frees the third chunk too, and keeps the second;
// the arena refuses a new allocation, and an ensure refuses for the partly freed arena without trying the teardown
// again. A later teardown frees the chunk that was kept and completes.
// Engagement: the arena has at least three chunks, so a chunk is freed before the failed one and one after it.
// Mutants: the free loop stopping at the failed chunk after one was freed (the third chunk is still found); the
// partly freed arena restored (it serves the allocation); the cause read as restorable (the ensure tries the
// teardown again, which completes).
static bool arena_teardown_partly_freed_stays_closed(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_partly_freed_stays_closed");
    constexpr size_t     mib       = 1024ull * 1024ull;
    const size_t         max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    unified_cache        cache(q, 128ull * mib, 0, 0, 0);
    arena_teardown_retry retry{ cache };
    if (!cache.arena_reserve(q, 128ull * mib, max_alloc, 48ull * mib, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 128 MiB arena, so no teardown was measured");
    }
    if (cache.chunk_count() < 3) {
        TEST_SKIP(skipped, "the fixture's arena has fewer than three chunks, so no teardown can fail between frees");
    }
    const std::vector<const void *> in_chunk = arena_pointer_in_each_chunk(cache);
    for (const void * p : in_chunk) {
        TEST_ASSERT(p != nullptr, "a chunk of the fixture's arena has no pointer the test could find");
    }

    unified_cache_test_fail_next_arena_free(1);
    TEST_ASSERT(!cache.arena_destroy(), "the teardown completed although the device refused to free a chunk");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::FREE_FAILED,
                "the teardown did not record the refused free after a chunk was freed");
    TEST_ASSERT(cache.arena_find_chunk(in_chunk[0]) < 0 && cache.arena_find_chunk(in_chunk[1]) == 1,
                "the teardown did not free the first chunk, or freed the chunk the device refused");
    for (size_t chunk = 2; chunk < in_chunk.size(); ++chunk) {
        TEST_ASSERT(cache.arena_find_chunk(in_chunk[chunk]) < 0,
                    "the teardown stopped at the refused chunk instead of freeing the chunks after it");
    }
    TEST_ASSERT(!arena_serves_a_new_allocation(cache), "the partly freed arena served a new allocation");
    zone_arena_refusal refused;
    TEST_ASSERT(!cache.ensure_planned_arena_zones(&refused) &&
                    refused.reason == zone_arena_refusal_reason::TEARDOWN_REFUSED &&
                    refused.teardown == zone_arena_teardown_refusal::FREE_FAILED,
                "an ensure did not refuse for the partly freed arena");
    TEST_ASSERT(cache.arena_find_chunk(in_chunk[1]) == 1, "the ensure tried the teardown of the partly freed arena");

    TEST_ASSERT(cache.arena_destroy(), "a later teardown did not free the chunk the refused one kept");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::NONE && !cache.arena_active(),
                "the completed teardown left the arena active or the earlier cause recorded");
    TEST_PASS();
    return true;
}

// A teardown that finds a lease on the first chunk it tries, which outlives the wait, frees nothing and leaves the
// arena as it was. The fixture holds a lease on the arena's only chunk; the lease is not an allocation, so the
// teardown reaches the chunk and waits for it, 100 ms here (the arm shortens the wait). The refusal naming
// LEASES_OUTSTANDING is the positive control, and no abort. Then the arena is as it was; once the lease is released,
// a second teardown completes.
// Mutants: the timed-out lease aborting the process, or the chunk freed under the lease (the teardown completes); the
// refusal not restoring (the SCRATCH zone refuses the allocation).
static bool arena_teardown_outstanding_lease_restores(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_outstanding_lease_restores");
    constexpr size_t       mib       = 1024ull * 1024ull;
    const size_t           max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    arena_lease_wait_short short_wait;
    unified_cache          cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry   retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no teardown was measured");
    }

    held_chunk_lease lease{ cache };
    lease.chunk = cache.arena_acquire_chunk_lease(cache.arena_base());
    TEST_ASSERT(lease.chunk == 0 && cache.chunk_count() == 1,
                "the fixture's lease is not on the only chunk of the arena");
    const arena_incarnations incarnations = arena_incarnations_of(cache);

    TEST_ASSERT(!cache.arena_destroy(), "the teardown completed although a lease held its only chunk");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::LEASES_OUTSTANDING,
                "the teardown did not record the outstanding lease as its cause");
    TEST_ASSERT(cache.arena_active() && cache.arena_find_chunk(cache.arena_base()) == 0,
                "the refused teardown freed the chunk the lease holds");
    TEST_ASSERT(arena_incarnations_unchanged(cache, incarnations),
                "a zone's authority or generation is not the one it had before the refused teardown");
    TEST_ASSERT(arena_serves_a_new_allocation(cache), "the arena refused a new allocation after the refused teardown");

    lease.release();
    TEST_ASSERT(cache.arena_destroy(), "a second teardown did not complete once the lease was released");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::NONE && !cache.arena_active(),
                "the completed teardown left the arena active or the earlier cause recorded");
    TEST_PASS();
    return true;
}

// A teardown that finds a lease on a later chunk, after it freed an earlier one, keeps the leased chunk and leaves the
// arena partly freed and closed: the path where the teardown used to abort. The arena is reserved in three chunks (a
// 48 MiB per-chunk cap under a 128 MiB budget), and the fixture holds a lease on the second. The teardown frees the
// first chunk, waits out the lease on the second (100 ms, as the arm shortens the wait), keeps it, and frees the
// third. The refusal naming LEASES_OUTSTANDING_AFTER_FREE is the positive control, and no abort. The arena refuses a
// new allocation. Then the lease is released, and an ensure still refuses for the partly freed arena without trying
// the teardown: the kept chunk is still there. A later teardown frees it and completes.
// Engagement: the arena has at least three chunks, so a chunk is freed before the leased one and one after it.
// Mutants: the leased chunk freed after its wait (the first teardown completes); the free loop stopping at the leased
// chunk (the third chunk is still found); the cause read as restorable (the ensure tries the teardown, which now
// completes, and does not refuse for the partly freed arena).
static bool arena_teardown_late_lease_stays_closed(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("arena_teardown_late_lease_stays_closed");
    constexpr size_t       mib       = 1024ull * 1024ull;
    const size_t           max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    arena_lease_wait_short short_wait;
    unified_cache          cache(q, 128ull * mib, 0, 0, 0);
    arena_teardown_retry   retry{ cache };
    if (!cache.arena_reserve(q, 128ull * mib, max_alloc, 48ull * mib, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 128 MiB arena, so no teardown was measured");
    }
    if (cache.chunk_count() < 3) {
        TEST_SKIP(skipped, "the fixture's arena has fewer than three chunks, so no lease can sit between frees");
    }
    const std::vector<const void *> in_chunk = arena_pointer_in_each_chunk(cache);
    for (const void * p : in_chunk) {
        TEST_ASSERT(p != nullptr, "a chunk of the fixture's arena has no pointer the test could find");
    }
    held_chunk_lease lease{ cache };
    lease.chunk = cache.arena_acquire_chunk_lease(in_chunk[1]);
    TEST_ASSERT(lease.chunk == 1, "the fixture's lease is not on the arena's second chunk");

    TEST_ASSERT(!cache.arena_destroy(), "the teardown completed although a lease held the arena's second chunk");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE,
                "the teardown did not record the lease that outlived its wait after a chunk was freed");
    TEST_ASSERT(cache.arena_find_chunk(in_chunk[0]) < 0 && cache.arena_find_chunk(in_chunk[1]) == 1,
                "the teardown did not free the first chunk, or freed the chunk the lease holds");
    for (size_t chunk = 2; chunk < in_chunk.size(); ++chunk) {
        TEST_ASSERT(cache.arena_find_chunk(in_chunk[chunk]) < 0,
                    "the teardown stopped at the leased chunk instead of freeing the chunks after it");
    }
    TEST_ASSERT(!arena_serves_a_new_allocation(cache), "the partly freed arena served a new allocation");

    lease.release();
    zone_arena_refusal refused;
    TEST_ASSERT(!cache.ensure_planned_arena_zones(&refused) &&
                    refused.reason == zone_arena_refusal_reason::TEARDOWN_REFUSED &&
                    refused.teardown == zone_arena_teardown_refusal::LEASES_OUTSTANDING_AFTER_FREE,
                "an ensure did not refuse for the partly freed arena");
    TEST_ASSERT(cache.arena_find_chunk(in_chunk[1]) == 1, "the ensure tried the teardown of the partly freed arena");

    TEST_ASSERT(cache.arena_destroy(), "a teardown after the lease ended did not free the chunk it held");
    TEST_ASSERT(cache.arena_teardown_refusal() == zone_arena_teardown_refusal::NONE && !cache.arena_active(),
                "the completed teardown left the arena active or the earlier cause recorded");
    TEST_PASS();
    return true;
}

// The arena teardown's refusal classes, each arm on a cache of its own. It runs alone in its process (`--case
// arena_teardown_refusal_classes`): its arms arm process-wide test hooks, and a refused teardown must not reach another
// case's cache. Exits 0 when every arm passed, 1 when one failed, and 77 when none failed but one measured nothing.
static int arena_teardown_refusal_classes(sycl::queue & q) {
    bool skipped = false;
    bool ok      = true;
    ok &= arena_teardown_live_allocations_restores(q, skipped);
    ok &= arena_teardown_first_free_refusal_restores(q, skipped);
    ok &= arena_teardown_drain_timeout_restores(q, skipped);
    ok &= arena_teardown_restore_finishes_an_interrupted_settle(q, skipped);
    ok &= arena_teardown_partly_freed_stays_closed(q, skipped);
    ok &= arena_teardown_outstanding_lease_restores(q, skipped);
    ok &= arena_teardown_late_lease_stays_closed(q, skipped);
    if (!ok) {
        return 1;
    }
    if (skipped) {
        fprintf(stderr, "SKIP: an arena teardown arm measured nothing on this device (its SKIPPED line says why)\n");
        return SYCL_TEST_SKIP;
    }
    return 0;
}

static std::mutex               g_staging_log_mutex;
static std::vector<std::string> g_staging_log_lines;

static void staging_log_capture(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    if (text == nullptr) {
        return;
    }
    fputs(text, stderr);
    std::lock_guard<std::mutex> lock(g_staging_log_mutex);
    g_staging_log_lines.emplace_back(text);
}

// How many of the lines logged while a cache asked for `staging_size` bytes of staging was built and destroyed name
// its staging buffer.
static size_t staging_lines_while_constructing(sycl::queue & q, size_t staging_size) {
    {
        std::lock_guard<std::mutex> lock(g_staging_log_mutex);
        g_staging_log_lines.clear();
    }

    struct log_restore {
        ~log_restore() { ggml_log_set(nullptr, nullptr); }
    } restore;

    ggml_log_set(staging_log_capture, nullptr);
    {
        unified_cache cache(q, 64ull * 1024ull * 1024ull, staging_size, 0, 0);
    }
    std::lock_guard<std::mutex> lock(g_staging_log_mutex);
    return static_cast<size_t>(
        std::count_if(g_staging_log_lines.begin(), g_staging_log_lines.end(), [](const std::string & line) {
            return line.find("unified_cache:staging") != std::string::npos ||
                   line.find("staging owner") != std::string::npos;
        }));
}

// A cache asked for no staging buffer, the size GGML_SYCL_HOST_STAGING_MB=0 asks for and the fixtures here pass,
// allocates none, so it logs no failed allocation of one. A zero-byte host allocation fails, and the cache logged that
// on every such construction. Engagement: a staging buffer of 1 PiB, past any host's RAM, is asked for first under the
// same capture, and its failure line must be seen. The driver aborted the process on that 1 PiB request instead of
// failing it, so this also pins that the raw host exit refuses a size past the RAM before the driver and the cache logs
// the failure. Mutants: the constructor asking the device for a zero-byte buffer; the raw host exit passing a size past
// the RAM to the driver (at 1 PiB the process aborted here).
static bool cache_without_staging_allocates_none(sycl::queue & q) {
    TEST_BEGIN("cache_without_staging_allocates_none");
    TEST_ASSERT(staging_lines_while_constructing(q, 1ull << 50) >= 1,
                "a staging buffer past the host's RAM failed without a line naming it, so this check cannot see one");
    TEST_ASSERT(staging_lines_while_constructing(q, 0) == 0,
                "a cache asked for no staging buffer asked for one anyway and logged that it failed");
    TEST_PASS();
    return true;
}

static bool host_zone_reset_trims_released_offload_pool_slots(sycl::queue & q) {
    TEST_BEGIN("host_zone_reset_trims_released_offload_pool_slots");
    offload_buffer_pool_trim(-1);

    constexpr size_t mib   = 1024ull * 1024ull;
    unified_cache *  cache = get_unified_cache(q);
    TEST_ASSERT(cache != nullptr, "cache unavailable");
    if (!cache->host_zones_configured()) {
        cache->configure_host_zones(4ull * mib, 4ull * mib, 16ull * mib, 4ull * mib);
    } else if (cache->host_zone_capacity(host_zone_id::STAGING) < mib) {
        TEST_ASSERT(cache->host_zone_grow(host_zone_id::STAGING, 16ull * mib), "failed to grow staging zone");
    }
    TEST_ASSERT(cache->host_zones_configured(), "host zones were not configured");
    TEST_ASSERT(cache->host_zone_capacity(host_zone_id::STAGING) >= mib, "staging zone too small");

    offload_buffer_request req{};
    req.queue                               = &q;
    req.device                              = -1;
    req.size                                = 4096;
    req.role                                = offload_buffer_role::SET_TENSOR_STAGE;
    req.intent.role                         = alloc_role::STAGING;
    req.intent.category                     = runtime_category::STAGING;
    req.intent.cohort_id                    = "test:host_zone_boundary_check";
    req.intent.constraints.must_host_pinned = true;

    offload_buffer_lease lease{};
    TEST_ASSERT(acquire_offload_buffer(req, &lease), "acquire failed");
    TEST_ASSERT(lease.valid && lease.handle.ptr != nullptr, "lease invalid");
    TEST_ASSERT(lease.handle.zone_managed, "expected zone-managed staging allocation");
    TEST_ASSERT(lease.handle.host_zone == host_zone_id::STAGING, "expected staging host-zone allocation");

    void *       ptr = lease.handle.ptr;
    alloc_metadata looked{};
    TEST_ASSERT(unified_lookup(ptr, &looked), "released lease should be registered before boundary check");
    TEST_ASSERT(release_offload_buffer(lease), "release failed");

    // host_zone_reset() before llama.cpp-37ba's rename. The offload pool
    // caches released leases for reuse rather than freeing them immediately
    // (see offload_buffer_pool_trim_host_zone(), called from inside
    // host_zone_settle()), so this call is still what purges an idle
    // offload-pool registration for STAGING, not a pure no-op liveness check.
    cache->host_zone_boundary_check(host_zone_id::STAGING);
    TEST_ASSERT(!unified_lookup(ptr, &looked), "boundary check should remove released offload-pool registration");

    TEST_PASS();
    return true;
}

// The planned terms of one device this case writes, read when it starts and written back when it ends, on every exit,
// through the getters and setters themselves rather than through the helper under test.
struct planned_terms_found {
    int                                device;
    size_t                             ring_weight;
    size_t                             ring_act;
    size_t                             ring_out;
    uint32_t                           ring_depth;
    uint32_t                           ring_n_ubatch;
    size_t                             act_row;
    size_t                             out_row;
    uint32_t                           n_expert;
    uint32_t                           n_expert_used;
    bool                               act_in_kv;
    bool                               out_in_kv;
    onednn_graph_scratch_planned_shape graph;
    nonfa_attn_scratch_planned_shape   nonfa;
    moe_control_requirement            control;
    size_t                             scatter;
    size_t                             moe_mmvq;

    explicit planned_terms_found(int d) :
        device(d),
        ring_weight(unified_cache_get_planned_pp_moe_onednn_weight_slot_bytes(d)),
        ring_act(unified_cache_get_planned_pp_moe_onednn_activation_slot_bytes(d)),
        ring_out(unified_cache_get_planned_pp_moe_onednn_output_slot_bytes(d)),
        ring_depth(unified_cache_get_planned_pp_moe_onednn_ring_depth(d)),
        ring_n_ubatch(unified_cache_get_planned_pp_moe_onednn_n_ubatch(d)),
        act_row(unified_cache_get_planned_pp_moe_onednn_activation_bytes_per_row(d)),
        out_row(unified_cache_get_planned_pp_moe_onednn_output_bytes_per_row(d)),
        n_expert(unified_cache_get_planned_pp_moe_onednn_n_expert(d)),
        n_expert_used(unified_cache_get_planned_pp_moe_onednn_n_expert_used(d)),
        act_in_kv(unified_cache_get_planned_pp_moe_onednn_activation_in_kv_zone(d)),
        out_in_kv(unified_cache_get_planned_pp_moe_onednn_output_in_kv_zone(d)),
        graph(unified_cache_get_planned_onednn_graph_scratch_shape(d)),
        nonfa(unified_cache_get_planned_nonfa_attn_scratch_shape(d)),
        control(unified_cache_get_planned_moe_control_requirement(d)),
        scatter(unified_cache_get_planned_moe_host_scatter_scratch_bytes(d)),
        moe_mmvq(unified_cache_get_planned_moe_mmvq_scratch_bytes(d)) {}

    ~planned_terms_found() {
        unified_cache_set_planned_pp_moe_onednn_scratch(device, ring_weight, ring_act, ring_out, ring_depth);
        unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, ring_n_ubatch);
        unified_cache_set_planned_pp_moe_onednn_row_bytes(device, act_row, out_row, n_expert, n_expert_used);
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, act_in_kv, out_in_kv);
        unified_cache_set_planned_onednn_graph_scratch_shape(device, graph.n_head_ctx_max, graph.n_head_swa_max,
                                                             graph.n_swa, graph.n_ubatch, graph.n_ctx);
        unified_cache_set_planned_nonfa_attn_scratch_shape(device, nonfa.n_head, nonfa.n_ubatch, nonfa.n_ctx);
        unified_cache_set_planned_moe_control_requirement(device, control);
        unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, scatter);
        unified_cache_set_planned_moe_mmvq_scratch_bytes(device, moe_mmvq);
    }
};

// A plan that is tried and discarded publishes nothing. The load stage's fallback and the plan dry run put back the
// terms the planner published while it computed the plan through planner_published_terms; this case checks that
// helper against the cache's own per-device state, and the RUNTIME requirement the zones are sized from with it.
// A load publishes its ring at the planning shape (n_ubatch 512) and its planner terms, a 4096-byte host scatter term
// among them: the requirement is R0. The snapshot is taken. The plan with the room raises the host scatter term by the
// run's 92160 bytes and publishes its own oneDNN Graph and non-FA shapes, and the micro-batch's publish grows the ring
// to n_ubatch 2048. The fallback publishes the ring at 512 again. Before the restore the requirement is R0 + 92160,
// what the zone was asked for when the run aborted, which shows the case reaches the term. After the restore it is
// exactly R0, and every planner term reads what the load published, the host scatter term its 4096 bytes and the MoE
// batched MMVQ bound the vehicle's 40-layer figure at 512 rows (the plan with the room published it at 2048 rows). A
// second leg moves the CONTROL requirement, the other planner term the RUNTIME requirement counts, and restores it the
// same way.
// Engagement: the ring's slots grow from 512 to 2048 rows, the requirement moves by exactly the term moved, and the
// MMVQ bound reads the discarded plan's figure before the restore.
// Mutants: a term left out of the snapshot or the restore (its leg reads the discarded plan's value); a restore that
// writes a constant, zero included (the load's nonzero shapes, host scatter, MMVQ and CONTROL bytes do not read back).
// No device work: the terms are per-device process state. The case leaves every term it writes as it found it.
static bool planner_terms_restore_returns_the_runtime_requirement() {
    TEST_BEGIN("planner_terms_restore_returns_the_runtime_requirement");
    constexpr int       device       = 0;
    constexpr size_t    mib          = 1024ull * 1024ull;
    constexpr size_t    scatter_term = 92160;
    constexpr size_t    load_scatter = 4096;
    // The vehicle's MoE batched MMVQ bound over 40 layers at 512 and at 2048 rows (test-zone-sizing Case 46).
    constexpr size_t    load_mmvq    = 190709760;
    constexpr size_t    wide_mmvq    = 762839040;
    planned_terms_found found(device);

    // The load: a ring of one 64 MiB weight slot whose activation and output slots follow the micro-batch, all of it in
    // the RUNTIME zone, and the planner's terms at the planning shape.
    unified_cache_set_planned_pp_moe_onednn_row_bytes(device, 5760, 11520, 32, 4);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, false, false);
    size_t act_512  = 0;
    size_t out_512  = 0;
    size_t act_2048 = 0;
    size_t out_2048 = 0;
    TEST_ASSERT(unified_cache_pp_moe_onednn_slots_for_ubatch(device, 512, &act_512, &out_512) &&
                    unified_cache_pp_moe_onednn_slots_for_ubatch(device, 2048, &act_2048, &out_2048),
                "the ring's slots cannot be sized from the row bytes");
    TEST_ASSERT(act_2048 > act_512 && out_2048 > out_512,
                "the ring's slots do not grow from 512 to 2048 rows, so the micro-batch's publish moves nothing");
    unified_cache_set_planned_pp_moe_onednn_scratch(device, 64 * mib, act_512, out_512, 1);
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 512);
    unified_cache_set_planned_onednn_graph_scratch_shape(device, 32, 0, 0, 512, 4096);
    unified_cache_set_planned_nonfa_attn_scratch_shape(device, 32, 512, 4096);
    unified_cache_set_planned_moe_control_requirement(device, moe_control_requirement{ mib, true });
    unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, load_scatter);
    unified_cache_set_planned_moe_mmvq_scratch_bytes(device, load_mmvq);
    size_t r0 = 0;
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(device, &r0),
                "the load's requirement is unanswerable");

    const planner_published_terms before = planner_published_terms::snapshot(device);

    // The plan with the room, and the micro-batch's publish.
    unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, load_scatter + scatter_term);
    unified_cache_set_planned_moe_mmvq_scratch_bytes(device, wide_mmvq);
    unified_cache_set_planned_onednn_graph_scratch_shape(device, 32, 0, 0, 2048, 4096);
    unified_cache_set_planned_nonfa_attn_scratch_shape(device, 32, 2048, 4096);
    unified_cache_set_planned_pp_moe_onednn_scratch(device, 64 * mib, act_2048, out_2048, 1);
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 2048);
    size_t grown = 0;
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(device, &grown) && grown > r0 + scatter_term,
                "the micro-batch's ring did not raise the requirement");

    // The fallback publishes the ring at the planning shape again; the discarded plan's term is still published.
    unified_cache_set_planned_pp_moe_onednn_scratch(device, 64 * mib, act_512, out_512, 1);
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 512);
    size_t unrestored = 0;
    TEST_ASSERT(
        unified_cache_get_planned_runtime_zone_requirement(device, &unrestored) && unrestored == r0 + scatter_term,
        "before the restore the requirement is not R0 + 92160, so the case does not reach the term");
    TEST_ASSERT(unified_cache_get_planned_moe_mmvq_scratch_bytes(device) == wide_mmvq,
                "before the restore the MMVQ bound is not the discarded plan's, so the case does not reach the term");

    before.restore(device);
    size_t restored = 0;
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(device, &restored) && restored == r0,
                "after the restore the requirement is not the load's R0");
    TEST_ASSERT(unified_cache_get_planned_moe_host_scatter_scratch_bytes(device) == load_scatter,
                "the restore did not put back the load's 4096-byte host scatter term");
    TEST_ASSERT(unified_cache_get_planned_moe_mmvq_scratch_bytes(device) == load_mmvq,
                "the restore left the discarded plan's MoE batched MMVQ bound published");
    const onednn_graph_scratch_planned_shape graph = unified_cache_get_planned_onednn_graph_scratch_shape(device);
    TEST_ASSERT(graph.n_head_ctx_max == 32 && graph.n_head_swa_max == 0 && graph.n_swa == 0 && graph.n_ubatch == 512 &&
                    graph.n_ctx == 4096,
                "the restore left the discarded plan's oneDNN Graph scratch shape published");
    const nonfa_attn_scratch_planned_shape nonfa = unified_cache_get_planned_nonfa_attn_scratch_shape(device);
    TEST_ASSERT(nonfa.n_head == 32 && nonfa.n_ubatch == 512 && nonfa.n_ctx == 4096,
                "the restore left the discarded plan's non-FA attention scratch shape published");

    // The CONTROL requirement: a discarded plan that moved it raises the requirement by what it moved, and the
    // restore puts the load's back.
    unified_cache_set_planned_moe_control_requirement(device, moe_control_requirement{ 2 * mib, true });
    size_t control_moved = 0;
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(device, &control_moved) && control_moved == r0 + mib,
                "a CONTROL requirement 1 MiB larger does not raise the requirement by 1 MiB");
    before.restore(device);
    const moe_control_requirement control = unified_cache_get_planned_moe_control_requirement(device);
    TEST_ASSERT(control.bytes == mib && control.valid, "the restore left the discarded plan's CONTROL requirement");
    TEST_ASSERT(unified_cache_get_planned_runtime_zone_requirement(device, &restored) && restored == r0,
                "after the CONTROL restore the requirement is not the load's R0");

    TEST_PASS();
    return true;
}

// The PP pipeline scratch and the oneDNN pair with its stored sum, which planned_terms_found does not keep, read when
// the case starts and written back when it ends.
struct pipeline_and_pair_found {
    int                  device;
    load_published_terms found;

    explicit pipeline_and_pair_found(int d) : device(d), found(load_published_terms::snapshot(d)) {}

    ~pipeline_and_pair_found() {
        unified_cache_set_planned_pp_pipeline_scratch_bytes(device, found.pp_pipeline);
        unified_cache_set_planned_onednn_scratchpad_pair(device, found.onednn_weights, found.onednn_activations);
        unified_cache_set_planned_onednn_scratchpad_bytes(device, found.onednn_stored);
    }
};

// A failed load puts back what it published and nothing another publisher wrote since: a runtime context transaction
// of a live model publishes some of the same terms while the load runs, and the oneDNN pair's reserve raises the stored
// sum outside the lock. The case drives load_published_record as the backend does: one record per load from the terms
// as the load's first stage found them, and a note per planner pass from the terms before and after it.
// The live model: pipeline 1 MiB, ring micro-batch 512 with no KV-zone slot, no host scatter, CONTROL 1 MiB, non-FA
// 32 heads x 512 x 4096, the pair 4 + 2 MiB with the sum its reserve raised to 8 MiB, the MoE batched MMVQ bound at its
// 512-row figure.
// Leg 1. Pass 1 of the load publishes all of those anew (the pair 16 + 8 MiB, so the sum 24 MiB; the MMVQ bound at
// 2048 rows). Between the passes a live context re-plans its ring (micro-batch 1024, its activation slot in the KV
// zone) and the reserve raises the sum to 30 MiB. Pass 2 publishes the ring and the pair again. After it a live context
// publishes its non-FA shape and raises the MMVQ bound to 1 GiB. The restore puts the pipeline, the scatter, CONTROL
// and the pair's halves back to the live model's; the ring to the context's re-plan and the sum to the reserve's
// raise, both of which the load wrote over; and it keeps the non-FA shape and the MMVQ bound the context published
// after the load.
// Leg 2. A load whose one pass publishes the pair and the MMVQ bound, after which the reserve raises the sum to 40 MiB:
// the restore keeps the raise and puts the halves and the live model's MMVQ bound back.
// Engagement: before each restore every term reads what the load or the later publisher wrote.
// Mutants: a restore of every term to what the load's first stage found (the non-FA shape reads 32 x 512, the ring 512,
// the sum 8 MiB, the MMVQ bound its 512-row figure); a restore of every group the load changed (the non-FA shape reads
// 32 x 512, the MMVQ bound its 512-row figure); a note that ignores another publisher (the ring reads 512, the sum of
// leg 1 8 MiB); a sum stored plainly (leg 2 reads 8 MiB); the MMVQ group left out of the restore (leg 2 reads the
// load's 2048-row figure).
// No device work: the terms are per-device process state. The case leaves every term it writes as it found it.
static bool load_record_keeps_a_later_publish() {
    TEST_BEGIN("load_record_keeps_a_later_publish");
    constexpr int           device    = 0;
    constexpr size_t        mib       = 1024ull * 1024ull;
    // The vehicle's MoE batched MMVQ bound over 40 layers at 512 and at 2048 rows (test-zone-sizing Case 46).
    constexpr size_t        live_mmvq = 190709760;
    constexpr size_t        load_mmvq = 762839040;
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);

    const auto live = [&]() {
        unified_cache_set_planned_pp_pipeline_scratch_bytes(device, mib);
        unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 512);
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, false, false);
        unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, 0);
        unified_cache_set_planned_moe_control_requirement(device, moe_control_requirement{ mib, true });
        unified_cache_set_planned_nonfa_attn_scratch_shape(device, 32, 512, 4096);
        unified_cache_set_planned_onednn_scratchpad_pair(device, 4 * mib, 2 * mib);
        unified_cache_set_planned_onednn_scratchpad_bytes(device, 8 * mib);
        unified_cache_set_planned_moe_mmvq_scratch_bytes(device, live_mmvq);
    };
    const auto load_pass = [&]() {
        unified_cache_set_planned_pp_pipeline_scratch_bytes(device, 3 * mib);
        unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 2048);
        unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, false, false);
        unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, 92160);
        unified_cache_set_planned_moe_control_requirement(device, moe_control_requirement{ 2 * mib, true });
        unified_cache_set_planned_nonfa_attn_scratch_shape(device, 32, 2048, 4096);
        unified_cache_set_planned_onednn_scratchpad_pair(device, 16 * mib, 8 * mib);
        unified_cache_set_planned_moe_mmvq_scratch_bytes(device, load_mmvq);
    };

    // Leg 1.
    live();
    load_published_record record(load_published_terms::snapshot(device));
    load_published_terms  before = load_published_terms::snapshot(device);
    load_pass();
    record.note(before, load_published_terms::snapshot(device));
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 1024);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, true, false);
    unified_cache_set_planned_onednn_scratchpad_bytes(device, 30 * mib);
    before = load_published_terms::snapshot(device);
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 2048);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, false, false);
    unified_cache_set_planned_onednn_scratchpad_pair(device, 16 * mib, 8 * mib);
    record.note(before, load_published_terms::snapshot(device));
    unified_cache_set_planned_nonfa_attn_scratch_shape(device, 64, 1024, 8192);
    TEST_ASSERT(unified_cache_raise_planned_moe_mmvq_scratch_bytes(device, 1024 * mib) == 1024 * mib,
                "a live context's raise of the MMVQ bound does not answer the raised figure");

    TEST_ASSERT(unified_cache_get_planned_pp_pipeline_scratch_bytes(device) == 3 * mib &&
                    unified_cache_get_planned_pp_moe_onednn_n_ubatch(device) == 2048 &&
                    unified_cache_get_planned_moe_host_scatter_scratch_bytes(device) == 92160 &&
                    unified_cache_get_planned_onednn_scratchpad_bytes_stored(device) == 24 * mib,
                "before the restore the load's pipeline, ring, scatter and sum do not read what it published");
    record.restore(device);
    const load_published_terms back = load_published_terms::snapshot(device);
    TEST_ASSERT(back.pp_pipeline == mib, "the restore left the failed load's pipeline scratch");
    TEST_ASSERT(back.planner.scatter == 0, "the restore left the failed load's host scatter term");
    TEST_ASSERT(back.planner.control.bytes == mib && back.planner.control.valid,
                "the restore left the failed load's CONTROL requirement");
    TEST_ASSERT(back.onednn_weights == 4 * mib && back.onednn_activations == 2 * mib,
                "the restore left the failed load's oneDNN pair");
    TEST_ASSERT(back.ring_n_ubatch == 1024 && back.ring_activation_in_kv && !back.ring_output_in_kv,
                "the ring is not the live context's re-plan the load wrote over");
    TEST_ASSERT(back.onednn_stored == 30 * mib, "the sum is not the reserve's raise the load wrote over");
    TEST_ASSERT(
        back.planner.nonfa.n_head == 64 && back.planner.nonfa.n_ubatch == 1024 && back.planner.nonfa.n_ctx == 8192,
        "the restore put back over the non-FA shape a live context published after the load");
    TEST_ASSERT(back.planner.moe_mmvq == 1024 * mib,
                "the restore put back over the MMVQ bound a live context raised after the load");

    // Leg 2.
    live();
    load_published_record raised(load_published_terms::snapshot(device));
    before = load_published_terms::snapshot(device);
    unified_cache_set_planned_onednn_scratchpad_pair(device, 16 * mib, 8 * mib);
    unified_cache_set_planned_moe_mmvq_scratch_bytes(device, load_mmvq);
    raised.note(before, load_published_terms::snapshot(device));
    unified_cache_set_planned_onednn_scratchpad_bytes(device, 40 * mib);
    TEST_ASSERT(unified_cache_get_planned_moe_mmvq_scratch_bytes(device) == load_mmvq,
                "before the restore the MMVQ bound does not read what the load published");
    raised.restore(device);
    const load_published_terms kept = load_published_terms::snapshot(device);
    TEST_ASSERT(kept.onednn_stored == 40 * mib, "the restore put back over the reserve's raise of the sum");
    TEST_ASSERT(kept.onednn_weights == 4 * mib && kept.onednn_activations == 2 * mib,
                "the restore left the failed load's oneDNN pair under a raised sum");
    TEST_ASSERT(kept.planner.moe_mmvq == live_mmvq, "the restore left the failed load's MMVQ bound");

    TEST_PASS();
    return true;
}

// The start values of every term a load publishes are a default load_published_terms, and its restore writes every
// group: the release of the last model puts each device it emptied back through load_published_terms{}.restore. In a
// process where no cache of any device has been made and nothing has been published, every device's snapshot equals
// the default. One value is then published in every group but the zone plan, which only a cache's own ensure stores,
// and the snapshot differs in exactly those groups; after load_published_terms{}.restore it equals the default again.
// Mutants: a term whose start value is not its default (the first check names the device); a group that restore()
// does not write (the last check reads its mask).
// No device work: the terms are per-device process state, and the case leaves them at their start values.
static bool fresh_published_terms_are_the_default(int device) {
    TEST_BEGIN("fresh_published_terms_are_the_default");
    constexpr size_t           mib = 1024ull * 1024ull;
    const load_published_terms start{};
    for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
        const uint32_t differs = load_published_terms::snapshot(d).differing(start);
        TEST_ASSERT(differs == 0, ("device " + std::to_string(d) + " reads group mask " + std::to_string(differs) +
                                   " away from a default load_published_terms before anything was published")
                                      .c_str());
    }

    unified_cache_set_planned_pp_pipeline_scratch_bytes(device, 3 * mib);
    unified_cache_set_planned_pp_moe_onednn_n_ubatch(device, 71);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, true, true);
    unified_cache_set_planned_pp_moe_onednn_row_bytes(device, 96, 192, 8, 2);
    unified_cache_set_planned_onednn_scratchpad_pair(device, 4 * mib, 2 * mib);
    unified_cache_set_planned_onednn_scratchpad_bytes(device, 8 * mib);
    unified_cache_set_planned_onednn_graph_scratch_shape(device, 8, 8, 0, 512, 4096);
    unified_cache_set_planned_nonfa_attn_scratch_shape(device, 32, 512, 4096);
    unified_cache_set_planned_moe_control_requirement(device, moe_control_requirement{ mib, true });
    unified_cache_set_planned_moe_host_scatter_scratch_bytes(device, 8192);
    unified_cache_set_planned_moe_mmvq_scratch_bytes(device, 16 * mib);
    const uint32_t published = load_published_terms::snapshot(device).differing(start);
    start.restore(device);
    const uint32_t restored = load_published_terms::snapshot(device).differing(start);

    const uint32_t want = LOAD_PUBLISHED_GROUP_ALL & ~static_cast<uint32_t>(LOAD_PUBLISHED_GROUP_ZONE_PLAN);
    TEST_ASSERT(published == want, ("one value in every group but the zone plan moved group mask " +
                                    std::to_string(published) + ", not " + std::to_string(want))
                                       .c_str());
    TEST_ASSERT(restored == 0, ("after load_published_terms{}.restore the groups of mask " + std::to_string(restored) +
                                " are not at their start values")
                                   .c_str());
    TEST_PASS();
    return true;
}

// Whether the cache's PP MoE oneDNN ring has a reserved slot 0. The read's copy holds the slot's owners, so it goes
// before this returns.
static bool ring_reserved(unified_cache & cache) {
    unified_cache::pp_moe_onednn_scratch_slot read;
    return cache.get_pp_moe_onednn_scratch_slot(0, read);
}

// Every byte the cache's VRAM zones count as used.
static size_t zone_bytes_used(unified_cache & cache) {
    size_t used = 0;
    for (size_t z = 0; z < static_cast<size_t>(vram_zone_id::COUNT); ++z) {
        used += cache.zone_used(static_cast<vram_zone_id>(z));
    }
    return used;
}

// The capacity of each of the cache's VRAM zones, by zone.
static std::vector<size_t> zone_capacities(unified_cache & cache) {
    std::vector<size_t> capacity(static_cast<size_t>(vram_zone_id::COUNT), 0);
    for (size_t z = 0; z < capacity.size(); ++z) {
        capacity[z] = cache.zone_capacity(static_cast<vram_zone_id>(z));
    }
    return capacity;
}

// The fixture's scratch on a hand-built cache, what a model's first prompt leaves: a one-slot PP MoE oneDNN ring of
// 1 MiB slots in the RUNTIME zone, planned on the cache's device first (the reserve admits only a planned ring), and a
// 1 + 1 MiB oneDNN pair from the ONEDNN zone. The caller's planned_terms_found and pipeline_and_pair_found put the
// device's terms back.
static bool reserve_ring_and_pair(unified_cache & cache, int device) {
    constexpr size_t mib = 1024ull * 1024ull;
    unified_cache_set_planned_pp_moe_onednn_scratch(device, mib, mib, mib, 1);
    unified_cache_set_planned_pp_moe_onednn_kv_zone_slots(device, false, false);
    return cache.reserve_pp_moe_onednn_scratch(mib, mib, mib, 1) && ring_reserved(cache) &&
           cache.reserve_onednn_scratch(mib, mib) && cache.has_onednn_scratch();
}

// The budget of a hand-built cache in which a load's start reserves the arena again (reserve_released_arena). That
// reserve builds the early arena the cache's constructor builds: the whole budget, with zones sized from the start
// terms, a 512 MiB SCRATCH zone and a 512 MiB RUNTIME zone (the GGML_SYCL_COMPUTE_ARENA_MB and
// GGML_SYCL_RUNTIME_ARENA_MB defaults) and a 256 MiB ONEDNN zone, beside a KV and weight zone of at least 16 MiB:
// 1296 MiB. Published terms only raise those zones, so a 64 MiB cache's start reserve is refused on every card, and
// an arm on one measures no start. The arms that score the start's reserve run it at the zones a real start builds,
// on a cache that holds them: 2 GiB of VRAM, held from the cache's construction to the arm's end (well under a
// second), except between a release and the reserve after it. A card that cannot reserve it skips the arm.
static constexpr size_t last_release_early_arena_budget = 2048ull * 1024ull * 1024ull;

static std::mutex                                          g_last_release_log_mutex;
static std::vector<std::pair<ggml_log_level, std::string>> g_last_release_log_lines;

static void last_release_log_capture(ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (text == nullptr) {
        return;
    }
    fputs(text, stderr);
    std::lock_guard<std::mutex> lock(g_last_release_log_mutex);
    g_last_release_log_lines.emplace_back(level, text);
}

// The lines logged, with their levels, while the guard lives.
struct last_release_log {
    last_release_log() {
        {
            std::lock_guard<std::mutex> lock(g_last_release_log_mutex);
            g_last_release_log_lines.clear();
        }
        ggml_log_set(last_release_log_capture, nullptr);
    }

    ~last_release_log() { ggml_log_set(nullptr, nullptr); }

    // How many captured lines at `level` contain `a`, and `b` too when it is given.
    size_t count(ggml_log_level level, const char * a, const char * b = nullptr) const {
        std::lock_guard<std::mutex> lock(g_last_release_log_mutex);
        size_t                      n = 0;
        for (const auto & [at, line] : g_last_release_log_lines) {
            n +=
                at == level && line.find(a) != std::string::npos && (b == nullptr || line.find(b) != std::string::npos);
        }
        return n;
    }
};

// A claimed slot of a cache's ring, released on every return. The claim's copy holds the slot's owners; they go before
// the slot is released, as a dispatch drops its copy before it releases its claim. Declared after the arm's
// arena_teardown_retry, so the claim ends before the retry's teardown.
struct claimed_ring_slot {
    unified_cache &                           cache;
    unified_cache::pp_moe_onednn_scratch_slot slot{};
    bool                                      held = false;

    ~claimed_ring_slot() { release(); }

    void release() {
        if (held) {
            const uint64_t generation = slot.generation;
            slot                      = {};
            cache.release_pp_moe_onednn_scratch_slot(0, generation);
            held = false;
        }
    }
};

// A reservation of a cache's oneDNN pair, as a matmul takes it, released on every return; declared after the arm's
// arena_teardown_retry.
struct reserved_onednn_pair {
    unified_cache &      cache;
    onednn_scratch_token token{};

    ~reserved_onednn_pair() { release(); }

    void release() {
        cache.release_onednn_scratch_reservation(token);
        token = {};
    }
};

// The release of the last model gives back what a cache sized for a model when no model is live, and nothing while one
// is. A hand-built cache holds the fixture's ring and pair in the early arena its constructor reserved
// (last_release_early_arena_budget). With a bit in its live mask the release returns MODEL_LIVE and the ring, the pair
// and the arena stay. With the mask 0 it returns EMPTIED: no ring slot reads back, the pair is gone, the arena is not
// active and no zone counts a used byte. A second release of the emptied cache returns EMPTIED again. A load's start
// then reserves the early arena again (reserve_released_arena): an active arena with nothing used in it.
// Engagement: the constructor reserved the early arena; the ring and the pair were reserved and the zones count their
// bytes.
// Mutants: the liveness check dropped (the first release empties the cache); a step left out of the release (the ring,
// the pair or the arena stays, or a zone counts bytes); a release that cannot run twice; a start that does not reserve
// the released arena again.
static bool last_release_empties_ring_pair_and_arena(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("last_release_empties_ring_pair_and_arena");
    const int               device = ggml_sycl_get_device_id_from_queue(q);
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);
    unified_cache           cache(q, last_release_early_arena_budget, 0, 0, 0);
    arena_teardown_retry    retry{ cache };
    if (!cache.arena_active()) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 2 GiB early arena, so no release was measured");
    }
    TEST_ASSERT(reserve_ring_and_pair(cache, device), "the fixture's ring or its oneDNN pair was not reserved");
    TEST_ASSERT(zone_bytes_used(cache) > 0, "the zones count no byte of the fixture's ring and pair");

    cache.set_live_model_mask(1u);
    const unified_cache_last_release live =
        cache.release_for_no_live_model("live-model", unified_cache_release_cause::MODEL_RELEASED);
    cache.set_live_model_mask(0u);
    TEST_ASSERT(live == unified_cache_last_release::MODEL_LIVE, "with a model live the release did not say so");
    TEST_ASSERT(ring_reserved(cache) && cache.has_onednn_scratch() && cache.arena_active(),
                "with a model live the release gave back the ring, the pair or the arena");

    const unified_cache_last_release emptied =
        cache.release_for_no_live_model("last-model", unified_cache_release_cause::MODEL_RELEASED);
    TEST_ASSERT(emptied == unified_cache_last_release::EMPTIED,
                "with no model live the release did not empty the cache");
    TEST_ASSERT(!ring_reserved(cache), "the PP MoE oneDNN ring stayed after the last release");
    TEST_ASSERT(!cache.has_onednn_scratch(), "the oneDNN pair stayed after the last release");
    TEST_ASSERT(!cache.arena_active(), "the arena stayed after the last release");
    TEST_ASSERT(zone_bytes_used(cache) == 0, "a zone counts used bytes after the last release");
    TEST_ASSERT(cache.release_for_no_live_model("again", unified_cache_release_cause::MODEL_RELEASED) ==
                    unified_cache_last_release::EMPTIED,
                "a second release of the emptied cache did not return EMPTIED");

    zone_arena_refusal refusal;
    TEST_ASSERT(cache.reserve_released_arena(&refusal),
                "a load's start did not reserve again the early arena the cache's constructor reserved");
    TEST_ASSERT(cache.arena_active() && zone_bytes_used(cache) == 0,
                "the arena a load's start reserved again is not active, or a zone counts used bytes in it");
    TEST_PASS();
    return true;
}

// A claimed slot of the ring keeps the ring and the arena: the release refuses with RING_CLAIMED and one ERROR line
// that says so, and gives back nothing, the pair included. Once the claim ends the release empties the cache.
// Engagement: slot 0 of the fixture's ring was claimed.
// Mutants: a claimed slot released (the ring is gone after the refused release); the release going on past the refusal
// (the pair or the arena is gone); the refusal silent (no ERROR line).
static bool last_release_keeps_a_claimed_ring(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("last_release_keeps_a_claimed_ring");
    constexpr size_t        mib       = 1024ull * 1024ull;
    const size_t            max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    const int               device    = ggml_sycl_get_device_id_from_queue(q);
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);
    unified_cache           cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry    retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no release was measured");
    }
    TEST_ASSERT(reserve_ring_and_pair(cache, device), "the fixture's ring or its oneDNN pair was not reserved");
    claimed_ring_slot claim{ cache };
    claim.held = cache.claim_pp_moe_onednn_scratch_slot(0, claim.slot);
    TEST_ASSERT(claim.held, "slot 0 of the fixture's ring was not claimed");

    unified_cache_last_release refused = unified_cache_last_release::FAILED;
    size_t                     errors  = 0;
    {
        const last_release_log log;
        refused = cache.release_for_no_live_model("claimed", unified_cache_release_cause::MODEL_RELEASED);
        errors  = log.count(GGML_LOG_LEVEL_ERROR, "[VRAM-ARENA] last release on device ",
                            "a slot of the PP MoE oneDNN ring is still claimed");
    }
    TEST_ASSERT(refused == unified_cache_last_release::RING_CLAIMED, "a claimed slot did not refuse the release");
    TEST_ASSERT(errors == 1, "the refused release did not log one ERROR line naming the claimed slot");
    TEST_ASSERT(ring_reserved(cache) && cache.has_onednn_scratch() && cache.arena_active(),
                "a release refused by a claimed slot gave back the ring, the pair or the arena");

    claim.release();
    TEST_ASSERT(cache.release_for_no_live_model("unclaimed", unified_cache_release_cause::MODEL_RELEASED) ==
                    unified_cache_last_release::EMPTIED,
                "once the claim ended the release did not empty the cache");
    TEST_ASSERT(!ring_reserved(cache) && !cache.arena_active(), "the ring or the arena stayed after the claim ended");
    TEST_PASS();
    return true;
}

// A reservation of the oneDNN pair keeps the pair and the arena: the release gives back the ring, then refuses with
// PAIR_HELD and one ERROR line that says so. Once the reservation ends the release empties the cache.
// Engagement: the fixture's pair was reserved as a matmul reserves it.
// Mutants: a reserved pair freed (the pair is gone after the refused release); the release going on past the refusal
// (the arena is gone); the refusal silent (no ERROR line).
static bool last_release_keeps_a_reserved_pair(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("last_release_keeps_a_reserved_pair");
    constexpr size_t        mib       = 1024ull * 1024ull;
    const size_t            max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    const int               device    = ggml_sycl_get_device_id_from_queue(q);
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);
    unified_cache           cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry    retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no release was measured");
    }
    TEST_ASSERT(reserve_ring_and_pair(cache, device), "the fixture's ring or its oneDNN pair was not reserved");
    reserved_onednn_pair                  pair{ cache };
    unified_cache::onednn_scratch_buffers buffers;
    TEST_ASSERT(cache.acquire_onednn_scratch_reservation(mib, mib, buffers, &pair.token),
                "the fixture's oneDNN pair was not reserved as a matmul reserves it");

    unified_cache_last_release refused = unified_cache_last_release::FAILED;
    size_t                     errors  = 0;
    {
        const last_release_log log;
        refused = cache.release_for_no_live_model("pair-held", unified_cache_release_cause::MODEL_RELEASED);
        errors  = log.count(GGML_LOG_LEVEL_ERROR, "[VRAM-ARENA] last release on device ",
                            "the oneDNN scratchpad pair is still reserved");
    }
    TEST_ASSERT(refused == unified_cache_last_release::PAIR_HELD, "a reserved pair did not refuse the release");
    TEST_ASSERT(errors == 1, "the refused release did not log one ERROR line naming the reserved pair");
    TEST_ASSERT(!ring_reserved(cache), "the ring, released before the pair, stayed");
    TEST_ASSERT(cache.has_onednn_scratch() && cache.arena_active(),
                "a release refused by a reserved pair gave back the pair or the arena");

    pair.release();
    TEST_ASSERT(cache.release_for_no_live_model("pair-free", unified_cache_release_cause::MODEL_RELEASED) ==
                    unified_cache_last_release::EMPTIED,
                "once the reservation ended the release did not empty the cache");
    TEST_ASSERT(!cache.has_onednn_scratch() && !cache.arena_active(),
                "the pair or the arena stayed after the reservation ended");
    TEST_PASS();
    return true;
}

// A lease on the arena's chunk keeps the arena, and the release names what holds it without reaping it. The ring and
// the pair go, then the release returns ARENA_HELD: at a failed load's abort with one WARN line, as the failed model's
// buffers go after it, and after a model's release with one ERROR line, as nothing should hold the arena then. Each
// line names the lease. Once the lease ends the release empties the cache.
// Engagement: the lease is on the arena's first chunk.
// Mutants: the lease reaped or the arena torn down under it (the arena is not active after the release); the holder
// not named (no line with chunk_leases=1); the levels swapped (an ERROR at the abort, a WARN after the release).
static bool last_release_names_a_held_arena(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("last_release_names_a_held_arena");
    constexpr size_t        mib       = 1024ull * 1024ull;
    const size_t            max_alloc = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    const int               device    = ggml_sycl_get_device_id_from_queue(q);
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);
    unified_cache           cache(q, 64ull * mib, 0, 0, 0);
    arena_teardown_retry    retry{ cache };
    if (!cache.arena_reserve(q, 64ull * mib, max_alloc, max_alloc, 8ull * mib, 8ull * mib, 8ull * mib, 0)) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 64 MiB arena, so no release was measured");
    }
    TEST_ASSERT(reserve_ring_and_pair(cache, device), "the fixture's ring or its oneDNN pair was not reserved");
    held_chunk_lease lease{ cache };
    lease.chunk = cache.arena_acquire_chunk_lease(cache.arena_base());
    TEST_ASSERT(lease.chunk >= 0, "the fixture's lease on the arena's first chunk was not taken");

    unified_cache_last_release aborted = unified_cache_last_release::FAILED;
    unified_cache_last_release after   = unified_cache_last_release::FAILED;
    size_t                     warned = 0, warned_as_error = 0, errored = 0, errored_as_warn = 0;
    {
        const last_release_log log;
        const char * const     abort_text   = "the failed load's buffers still hold the arena";
        const char * const     release_text = "no model is live but the arena is held";
        aborted         = cache.release_for_no_live_model("aborted", unified_cache_release_cause::LOAD_ABORTED);
        warned          = log.count(GGML_LOG_LEVEL_WARN, abort_text, "chunk_leases=1");
        warned_as_error = log.count(GGML_LOG_LEVEL_ERROR, abort_text);
        after           = cache.release_for_no_live_model("released", unified_cache_release_cause::MODEL_RELEASED);
        errored         = log.count(GGML_LOG_LEVEL_ERROR, release_text, "chunk_leases=1");
        errored_as_warn = log.count(GGML_LOG_LEVEL_WARN, release_text);
    }
    TEST_ASSERT(aborted == unified_cache_last_release::ARENA_HELD,
                "at an abort a held arena did not refuse the release");
    TEST_ASSERT(warned == 1 && warned_as_error == 0, "the abort's release did not log one WARN line naming the lease");
    TEST_ASSERT(after == unified_cache_last_release::ARENA_HELD, "after a release a held arena did not refuse it");
    TEST_ASSERT(errored == 1 && errored_as_warn == 0, "the model release did not log one ERROR line naming the lease");
    TEST_ASSERT(!ring_reserved(cache) && !cache.has_onednn_scratch(), "the ring or the pair stayed in a held arena");
    TEST_ASSERT(cache.arena_active() && cache.arena_find_chunk(cache.arena_base()) == lease.chunk,
                "the release tore down or freed the chunk a lease holds");

    lease.release();
    TEST_ASSERT(cache.release_for_no_live_model("lease-ended", unified_cache_release_cause::MODEL_RELEASED) ==
                    unified_cache_last_release::EMPTIED,
                "once the lease ended the release did not empty the cache");
    TEST_ASSERT(!cache.arena_active(), "the arena stayed after the lease ended");
    TEST_PASS();
    return true;
}

// A failed load whose buffers hold the arena at its abort defers the release to the next load's start, which runs it
// again and, once nothing holds the arena, starts from the zones a last model's release leaves. The hand-built cache's
// constructor reserves the early arena (last_release_early_arena_budget). The reference: a model's release empties the
// cache and a load's start reserve builds the early arena again, at the constructor's zones, which every later start
// must match. Then a lease on that arena's first chunk holds it. A load's start that finds it held on a
// cache with no mark keeps the arena and sets no mark: only the abort sets it. Through a failed load's abort the lease
// gives ARENA_HELD, and the release is marked deferred. A load's start while the lease still holds the arena runs the
// release again, as ggml_backend_sycl_model_load_begin does for a marked cache: ARENA_HELD with one WARN line that says
// the load runs on the arena as it is and that a holder left once the failed model is freed is a leak, the holders
// named on a WARN line too and nothing at ERROR; the mark stays, the reserve does nothing and the chunk is still
// leased. Once the lease ends, a load's start empties the cache, clears the mark and reserves the early arena again at
// the reference zones, with nothing used in them.
// Engagement: the constructor reserved the early arena, the reference reserve built it again, and the lease is on its
// first chunk.
// Mutants: the mark not set at the abort, or set at a start; the mark cleared by a release that kept the arena; the
// mark kept after the cache emptied; the start's held arena logged with the abort's text, at ERROR or without its
// holders; a start that rebuilds other zones than a last release's (the zones differ from the reference) or than the
// constructor's (the reference differs from them).
static bool last_release_deferred_by_a_failed_load(sycl::queue & q, bool & skipped) {
    TEST_BEGIN("last_release_deferred_by_a_failed_load");
    const int               device = ggml_sycl_get_device_id_from_queue(q);
    planned_terms_found     found(device);
    pipeline_and_pair_found found_pair(device);
    unified_cache           cache(q, last_release_early_arena_budget, 0, 0, 0);
    arena_teardown_retry    retry{ cache };
    if (!cache.arena_active()) {
        TEST_SKIP(skipped, "the device did not reserve the fixture's 2 GiB early arena, so no start was compared");
    }
    const std::vector<size_t> built_zones = zone_capacities(cache);
    TEST_ASSERT(cache.release_for_no_live_model("reference", unified_cache_release_cause::MODEL_RELEASED) ==
                    unified_cache_last_release::EMPTIED,
                "the reference's model release did not empty the cache");
    zone_arena_refusal refusal;
    TEST_ASSERT(cache.reserve_released_arena(&refusal),
                "a load's start did not reserve again the early arena the cache's constructor reserved");
    const std::vector<size_t> first_zones = zone_capacities(cache);
    TEST_ASSERT(first_zones == built_zones, "a load's start built other zones than the cache's constructor built");
    TEST_ASSERT(!cache.last_release_deferred(), "a model's release that emptied the cache left a deferred mark");

    held_chunk_lease lease{ cache };
    lease.chunk = cache.arena_acquire_chunk_lease(cache.arena_base());
    TEST_ASSERT(lease.chunk >= 0, "the fixture's lease on the early arena's first chunk was not taken");
    TEST_ASSERT(cache.release_for_no_live_model("unmarked-start", unified_cache_release_cause::LOAD_START) ==
                    unified_cache_last_release::ARENA_HELD,
                "a load's start released an arena a lease holds");
    TEST_ASSERT(!cache.last_release_deferred(), "a load's start marked a release deferred: only the abort marks it");
    TEST_ASSERT(cache.release_for_no_live_model("aborted", unified_cache_release_cause::LOAD_ABORTED) ==
                    unified_cache_last_release::ARENA_HELD,
                "at an abort a held arena did not refuse the release");
    TEST_ASSERT(cache.last_release_deferred(), "a failed load's held arena did not mark its release deferred");

    unified_cache_last_release held_start = unified_cache_last_release::FAILED;
    bool                       held_kept  = false;
    size_t                     warned = 0, holders_named = 0, abort_text = 0, errored = 0;
    {
        const last_release_log log;
        held_start = cache.release_for_no_live_model("load-start", unified_cache_release_cause::LOAD_START);
        held_kept  = cache.reserve_released_arena(&refusal);
        warned = log.count(GGML_LOG_LEVEL_WARN, "(load-start): a load is starting on an arena a failed load left held",
                           "a holder still present after that failed model was freed is a leak");
        holders_named = log.count(GGML_LOG_LEVEL_WARN, "[VRAM-ARENA] live in the arena on device");
        abort_text    = log.count(GGML_LOG_LEVEL_WARN, "the failed load's buffers still hold the arena");
        errored       = log.count(GGML_LOG_LEVEL_ERROR, "[VRAM-ARENA]");
    }
    TEST_ASSERT(held_start == unified_cache_last_release::ARENA_HELD,
                "a load's start released an arena the failed load's buffers still hold");
    TEST_ASSERT(warned == 1 && abort_text == 0,
                "the start's release did not log its one WARN line saying a holder left after the free is a leak");
    TEST_ASSERT(holders_named == 1 && errored == 0,
                "the start's release did not name the holders at WARN, or logged an ERROR line");
    TEST_ASSERT(cache.last_release_deferred(), "a release that kept the arena cleared the deferred mark");
    TEST_ASSERT(held_kept && cache.arena_active() && cache.arena_find_chunk(cache.arena_base()) == lease.chunk,
                "a load's start on the held arena refused, tore it down or freed the leased chunk");

    lease.release();
    const unified_cache_last_release free_start =
        cache.release_for_no_live_model("load-start", unified_cache_release_cause::LOAD_START);
    TEST_ASSERT(free_start == unified_cache_last_release::EMPTIED,
                "once the lease ended a load's start did not empty the cache");
    TEST_ASSERT(!cache.last_release_deferred(), "the deferred mark stayed after the cache emptied");
    TEST_ASSERT(cache.reserve_released_arena(&refusal), "a load's start did not reserve the early arena again");
    TEST_ASSERT(zone_capacities(cache) == first_zones,
                "a load's start after the deferred release built other zones than a last model's release leaves");
    TEST_ASSERT(zone_bytes_used(cache) == 0, "a zone counts used bytes in the early arena the load's start reserved");
    TEST_PASS();
    return true;
}

// The release of the last model, each arm on a cache of its own, and the start values it puts a device's published
// terms back to. Its own process (`--case last_release_empties_the_arena`): the terms arm must read a process where
// nothing has been published, and the arms publish the device's ring and pair terms. Exits 0 when every arm passed, 1
// when one failed, and 77 when none failed but one measured nothing.
static int last_release_empties_the_arena(sycl::queue & q) {
    bool skipped = false;
    bool ok      = true;
    ok &= fresh_published_terms_are_the_default(ggml_sycl_get_device_id_from_queue(q));
    ok &= last_release_empties_ring_pair_and_arena(q, skipped);
    ok &= last_release_keeps_a_claimed_ring(q, skipped);
    ok &= last_release_keeps_a_reserved_pair(q, skipped);
    ok &= last_release_names_a_held_arena(q, skipped);
    ok &= last_release_deferred_by_a_failed_load(q, skipped);
    if (!ok) {
        return 1;
    }
    if (skipped) {
        fprintf(stderr, "SKIP: a last-release arm measured nothing on this device (its SKIPPED line says why)\n");
        return SYCL_TEST_SKIP;
    }
    return 0;
}

int main(int argc, char ** argv) {
    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Unified Runtime Allocator Tests\n");
    fprintf(stderr, "===========================================\n");

    const bool host_inventory_case = argc == 3 && std::strcmp(argv[1], "--case") == 0 &&
                                     std::strcmp(argv[2], "host_inventory_initializes_zones") == 0;
    if (host_inventory_case) {
        // Before any SYCL/backend initialization or memoized env reads. Use
        // controls already implemented at the historical RED source SHA;
        // GGML_SYCL_HOST_RESERVE_MB was dead there, so do not depend on it.
        set_env_var("GGML_SYCL_VRAM_BUDGET_PCT", nullptr);
        set_env_var("GGML_SYCL_VRAM_ARENA_EXTERNAL_HEADROOM_MB", nullptr);
        set_env_var("GGML_SYCL_COMPUTE_ARENA_MB", "16");
        set_env_var("GGML_SYCL_RUNTIME_ARENA_MB", "16");
        set_env_var("GGML_SYCL_S1_MAX_IN_FLIGHT", "1");
        set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");
        set_env_var("GGML_SYCL_HOST_STAGING_MB", "1");
        set_env_var("GGML_SYCL_HOST_RESERVE_MB", nullptr);
        set_env_var("GGML_SYCL_MOE_MULTI_GPU", "0");
        set_env_var("GGML_SYCL_FORCE_STREAMING", "0");
        ggml_backend_sycl_set_unified_cache_host_budget_pct(1);
    }

    if (std::getenv("GGML_SYCL_PINNED_CHUNK_MB") == nullptr) {
        set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");
    }

#if GGML_SYCL_DNNL
    // llama.cpp-me60 F1: both memoize their env var in a function-local
    // static on first read (onednn_graph_scratch_direct_cap_bytes(),
    // onednn_graph_scratch_test_hooks_enabled() in unified-cache.cpp), so
    // setting either later would be a silent no-op -- nothing before this
    // point in main() touches the oneDNN Graph-scratch subsystem, so this
    // is genuinely before any such first read.
    //
    // CAP: 700 MB, not the default -- raised so
    // onednn_graph_scratch_flag_slab_released_at_module_shutdown()'s 300 MiB
    // DIRECT request (see that function's own comment for why it must
    // exceed the ~256 MB no-model zone floor) has headroom under the cap
    // regardless of what this process's own arena planning leaves the
    // default at; mirrors tests/test-sycl-onednn-graph-scratch-direct.cpp's
    // own kCapMiB rationale.
    set_env_var("GGML_SYCL_ONEDNN_GRAPH_DIRECT_CAP_MB", "700");
    // TEST HOOKS: without this,
    // onednn_graph_scratch_pool_entry_flag_true_for_test() returns false
    // unconditionally (see its own comment in unified-cache.hpp) and the
    // new case's poll loop would just run out its deadline every time.
    set_env_var("GGML_SYCL_ONEDNN_GRAPH_TEST_HOOKS", "1");
#endif

    // Select a device BEFORE constructing the queue. The bare `sycl::queue q;`
    // this replaced default-constructs through the default selector, which THROWS
    // on a device-less host -- from outside the try, so the process aborted (exit
    // 134) without ever reaching either fallback. See sycl-test-skip.hpp. The
    // GPU-preferred / any-device-accepted intent is unchanged; only the exit code
    // for "no device at all" moves from 1 to 77, so a CPU-only runner reports a
    // skip rather than a hard failure.
    std::optional<sycl::device> dev_opt = sycl_test_prefer_gpu("the unified runtime allocator");
    if (!dev_opt) {
        return SYCL_TEST_SKIP;
    }
    sycl::device & dev = *dev_opt;
    sycl::queue q(dev, sycl::property::queue::in_order{});

    if (host_inventory_case) {
        bool ok = false;
        try {
            ok = host_inventory_initializes_zones(q);
        } catch (const std::exception & e) {
            fprintf(stderr, "host inventory fixture exception: %s\n", e.what());
        }
        if (!shutdown_unified_cache()) {
            fprintf(stderr, "host inventory fixture shutdown failed\n");
            ok = false;
        }
        fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
        return ok ? 0 : 1;
    }

    // This child intentionally omits explicit shutdown. Its ordinary return
    // exercises the production exit path, where the per-device cache map
    // destroys its cache after main. The map is made at its first use, so the
    // cache comes first and the zone-sizing diagnostics after it, the order of
    // the historical destructor-order crash. Today the diagnostics' state is
    // leaked on purpose and nothing of it is destroyed at exit; the order
    // matters only if that state is given a destructor again, which would then
    // run before the cache's teardown reports into it.
    if (argc == 2 && std::strcmp(argv[1], "--static-destruction-child") == 0) {
        auto * cache = get_unified_cache(q);
        zone_sizing_record_observation("static-destruction-child");
        return cache ? 0 : 1;
    }

    // llama.cpp-me60: run exactly ONE of the two shutdown-poisoning cases,
    // in its OWN process, so a refused shutdown_unified_cache() call from
    // one never carries over into the other. Measured on hardware (see
    // onednn_graph_scratch_flag_slab_released_at_module_shutdown()'s own
    // comment): a refused shutdown is not retryable-safe for the oneDNN
    // Graph-scratch DIRECT pool -- it leaves the cache partially torn down,
    // and a later onednn_graph_scratch_alloc() on that same cache resolved
    // off-device and hit GGML_ABORT. Two ctest entries of THIS SAME binary
    // (same ENVIRONMENT/LABELS/TIMEOUT as the default registration, names
    // suffixed with the case) invoke this with `--case <name>` so each is a
    // valid, independent regression witness; the plain no-argument run
    // below does NOT call either case -- it prints one line naming them and
    // why (see that block's own comment): any completed
    // shutdown_unified_cache() call earlier in the process poisons a
    // lazily recreated cache, so a default-run invocation would exercise a
    // test-only defect rather than either case's own target property.
    if (argc >= 2 && std::strcmp(argv[1], "--case") == 0) {
        // Checked BEFORE any argv[2] use below (both the GGML_SYCL_DNNL
        // dispatch and its #else SKIP branch dereference argv[2]) -- an
        // arity check folded into the outer `if` alongside the flag match
        // (the form this replaced) fails OPEN on wrong arity: `--case`
        // with no name at all does not match `argc == 3`, so it fell
        // through silently to the full default suite below instead of
        // being reported as a usage error.
        if (argc != 3) {
            fprintf(stderr, "usage: %s --case <name>\n", argv[0]);
            return 1;
        }
        // llama.cpp-nsl3: pool-level, no oneDNN dependency, so it dispatches
        // before the GGML_SYCL_DNNL block and runs on any SYCL device
        // (including the CPU device) -- the only GPU-bound input is the
        // pinned_chunk_pool's queue.
        if (std::strcmp(argv[2], "host_zone_grow_serves_oversize_contiguous_alloc") == 0) {
            const bool case_ok = host_zone_grow_serves_oversize_contiguous_alloc(q);
            fprintf(stderr, "-------------------------------------------\n");
            fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
            return case_ok ? 0 : 1;
        }
        if (std::strcmp(argv[2], "host_zone_grow_serves_odd_residue_oversize_alloc") == 0) {
            const bool case_ok = host_zone_grow_serves_odd_residue_oversize_alloc(q);
            fprintf(stderr, "-------------------------------------------\n");
            fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
            return case_ok ? 0 : 1;
        }
        // No oneDNN dependency either; it allocates its own small arenas on the device.
        if (std::strcmp(argv[2], "arena_teardown_refusal_classes") == 0) {
            const int case_rc = arena_teardown_refusal_classes(q);
            fprintf(stderr, "-------------------------------------------\n");
            fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
            return case_rc;
        }
        // No oneDNN dependency; its first arm needs a process where nothing has been published.
        if (std::strcmp(argv[2], "last_release_empties_the_arena") == 0) {
            const int case_rc = last_release_empties_the_arena(q);
            fprintf(stderr, "-------------------------------------------\n");
            fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
            return case_rc;
        }
#if GGML_SYCL_DNNL
        const char * case_name = argv[2];
        bool         case_ok;
        if (std::strcmp(case_name, "onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim") == 0) {
            case_ok = onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim(q);
        } else if (std::strcmp(case_name, "onednn_graph_scratch_flag_slab_released_at_module_shutdown") == 0) {
            case_ok = onednn_graph_scratch_flag_slab_released_at_module_shutdown(q);
        } else {
            fprintf(stderr, "unknown --case %s\n", case_name);
            return 1;
        }
        fprintf(stderr, "-------------------------------------------\n");
        fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
        return case_ok ? 0 : 1;
#else
        // The two named cases only exist under GGML_SYCL_DNNL (the whole
        // onednn_graph_scratch_* subsystem is guarded the same way) --
        // consistent with this binary's other GGML_SYCL_DNNL-gated
        // behavior, report a real SKIP (77) rather than silently falling
        // through to the full default suite below.
        fprintf(stderr, "SKIP: --case %s requires GGML_SYCL_DNNL, not enabled in this build\n", argv[2]);
        return SYCL_TEST_SKIP;
#endif
    }

    bool ok = true;
    enable_strict_mode_env();
    ok &= reserve_allocate_success_registers_pointer(q);
    ok &= arena_registry_commit_failure_rolls_back(q);
    ok &= allocate_failure_rolls_back_budget(q);
    ok &= raw_gate_refuses_past_the_device_total(q);
    ok &= free_unknown_pointer_fails();
    ok &= strict_unknown_free_fails();
    ok &= double_free_fails(q);
    ok &= lookup_returns_correct_metadata(q);
    ok &= cohort_prefers_weight_tier_for_compute(q);
    ok &= hard_constraint_overrides_cohort(q);
    ok &= policy_never_selects_shared_usm(q);
    ok &= strict_stale_handle_fails(q);
    ok &= strict_device_mismatch_fails(q);
    ok &= scoped_unified_alloc_frees_on_scope_exit(q);
    ok &= offload_pool_reuse_tracks_hit_miss(q);
    ok &= offload_pool_stale_lease_fails(q);
    ok &= offload_pool_trim_clears_released_entries(q);
    ok &= offload_pool_set_tensor_roles_reuse(q);
    ok &= offload_wait_stats_split_tracks_forced_and_fallback();
    ok &= offload_cross_domain_stats_split_by_phase();
    ok &= offload_transfer_bytes_split_by_phase();
    ok &= offload_dispatch_counts_split_by_phase();
    ok &= offload_phase_roundtrip();
    ok &= offload_transition_wait_stats_split_by_phase();
    ok &= offload_host_alloc_stats_split_by_tag();
    ok &= offload_raw_alloc_and_fallback_stats_are_counted();
    ok &= direct_stage_host_fallback_counts_attempt(q);
    ok &= host_zone_contiguous_alloc_skips_chunk_tail(q);
    ok &= host_zone_config_ordering_matters(q);
    ok &= host_zone_grow_serves_oversize_contiguous_alloc(q);
    ok &= host_zone_grow_serves_odd_residue_oversize_alloc(q);
    ok &= host_zone_reset_trims_released_offload_pool_slots(q);
    ok &= independent_exact_token_defers_owned_release(q);
    ok &= hand_built_arena_is_judged_by_its_own_zone_plan(q);
    ok &= arena_owned_shutdown_and_lifecycle_serialization(q);
    ok &= arena_shutdown_drains_dma_and_bcs(q);
    ok &= exact_scratch_shutdown_record_survives_for_retry(q);
    ok &= scratch_regrow_refusal_preserves_exact_record(q);
    ok &= concurrent_settle_destroy_closing_wins(q);
    // The refused-teardown arms arm process-wide failure hooks, so they run in a process of their own.
    fprintf(stderr, "(the arena teardown refusal arms run only via --case arena_teardown_refusal_classes)\n");
    // The last-release arms read a process where nothing has been published, so they too run in a process of their own.
    fprintf(stderr, "(the last-release arms run only via --case last_release_empties_the_arena)\n");
    ok &= cache_without_staging_allocates_none(q);
    ok &= accepted_control_publication_is_atomic_with_close_snapshot();
    ok &= planner_terms_restore_returns_the_runtime_requirement();
    ok &= load_record_keeps_a_later_publish();
    ok &= global_cache_static_destruction_exits_cleanly(argv[0]);
    // Retain a real pinned suballocation across the first shutdown attempt;
    // retry drains the process-global cache while q and SYCL are still alive.
    ok &= retained_pinned_suballocation_refuses_preteardown(q);
    ok &= explicit_global_cache_shutdown_is_clean();
#if GGML_SYCL_DNNL
    // llama.cpp-me60: deliberately NOT run here. Both shutdown-poisoning
    // cases (onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim,
    // onednn_graph_scratch_flag_slab_released_at_module_shutdown) require a
    // cache that has NEVER been through a completed shutdown_unified_cache()
    // call anywhere earlier in this process -- see either case's own
    // comment for the verified mechanism (g_sycl_shutting_down staying set
    // across a lazily-recreated cache). This default run has already run
    // several such completed shutdowns above
    // (retained_pinned_suballocation_refuses_preteardown,
    // explicit_global_cache_shutdown_is_clean), so running either case here
    // would exercise the SAME test-only defect their own comments describe,
    // not the property either case actually targets. They run ONLY through
    // their own ctest registrations
    // (sycl-runtime-alloc-flag-slab-survives-after-pool-reclaim,
    // sycl-runtime-alloc-flag-slab-released-at-shutdown), each `--case
    // <name>` in a fresh process with nothing run before it.
    fprintf(stderr,
            "(skipping onednn_graph_scratch_flag_slab_survives_module_shutdown_after_pool_reclaim and "
            "onednn_graph_scratch_flag_slab_released_at_module_shutdown here -- they run only via their own "
            "--case ctest registrations, in a fresh process)\n");
#endif

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
    return ok ? 0 : 1;
}
