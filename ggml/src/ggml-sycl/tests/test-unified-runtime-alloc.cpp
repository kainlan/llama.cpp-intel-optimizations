//
// Unified runtime allocator tests
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../unified-cache.hpp"
#include "../ggml-sycl-test.hpp"
#include "../zone-sizing.hpp"

#include "sycl-test-skip.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <sycl/sycl.hpp>

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
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 4096;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "unified_alloc failed");
    TEST_ASSERT(h.ptr != nullptr, "allocated pointer is null");

    alloc_handle looked{};
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

    alloc_handle retry{};
    TEST_ASSERT(unified_alloc(req, &retry), "allocator did not recover after publication rollback");
    TEST_ASSERT(retry.alloc_id != 0, "retry omitted exact allocation identity");
    TEST_ASSERT(unified_free(retry), "retry cleanup failed");
    TEST_PASS();
    return true;
}

static bool allocate_failure_rolls_back_budget(sycl::queue & q) {
    TEST_BEGIN("allocate_failure_rolls_back_budget");
    const int    device = 0;
    const size_t before = unified_cache_arena_non_weight_used(device);

    alloc_request req;
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
    alloc_request req;
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
    alloc_request req;
    req.queue                               = &q;
    req.size                                = 8192;
    req.intent.role                         = alloc_role::COMPUTE;
    req.intent.category                     = runtime_category::COMPUTE;
    req.intent.constraints.must_host_pinned = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    alloc_handle looked{};
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
    alloc_request seed;
    seed.queue                               = &q;
    seed.size                                = 4096;
    seed.intent.role                         = alloc_role::WEIGHT;
    seed.intent.category                     = runtime_category::OTHER;
    seed.intent.cohort_id                    = "test:cohort";
    seed.intent.constraints.must_host_pinned = true;

    alloc_handle seed_h{};
    TEST_ASSERT(unified_alloc(seed, &seed_h), "seed alloc failed");

    alloc_request req;
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
    alloc_request req;
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
    alloc_request req;
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
    alloc_request req;
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
    alloc_request req;
    req.queue                          = &q;
    req.size                           = 1024;
    req.intent.role                    = alloc_role::COMPUTE;
    req.intent.category                = runtime_category::COMPUTE;
    req.intent.constraints.must_device = true;

    alloc_handle h{};
    TEST_ASSERT(unified_alloc(req, &h), "alloc failed");
    TEST_ASSERT(!unified_free_ptr(h.ptr, h.device + 1), "device mismatch free should fail");
    alloc_handle looked{};
    TEST_ASSERT(unified_lookup(h.ptr, &looked), "allocation should remain registered after mismatch");
    TEST_ASSERT(unified_free(h), "cleanup free failed");
    TEST_PASS();
    return true;
}

static bool scoped_unified_alloc_frees_on_scope_exit(sycl::queue & q) {
    TEST_BEGIN("scoped_unified_alloc_frees_on_scope_exit");
    alloc_request req;
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
        alloc_handle looked{};
        TEST_ASSERT(unified_lookup(ptr, &looked), "lookup should succeed while in scope");
    }
    alloc_handle looked{};
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

    constexpr size_t src_size = 4096;
    void *           src      = sycl::malloc_host(src_size, q);
    TEST_ASSERT(src != nullptr, "failed to allocate tiny host source");
    std::memset(src, 0x5a, src_size);

    static int key_tag;
    const ggml_sycl_cache_id key      = test_make_cache_id(&key_tag, 0xfeed);
    const size_t             dst_size = (size_t) 1 << 50;

    offload_stats_reset();
    const direct_stage_result result = cache->direct_stage_expert(key, src, src_size, dst_size, GGML_LAYOUT_SOA,
                                                                  nullptr, nullptr, &q);
    const offload_stats_snapshot stats = offload_stats_get();
    const size_t dropped = cache->drop_expert_entries_for_tensor_layout(
        std::vector<ggml_sycl_cache_id>{ key }, GGML_LAYOUT_AOS, "test-direct-stage-host-fallback-counts-attempt");
    sycl::free(src, q);

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

static bool explicit_global_cache_shutdown_is_clean() {
    TEST_BEGIN("explicit_global_cache_shutdown_is_clean");
    TEST_ASSERT(shutdown_unified_cache(), "explicit global cache shutdown failed");
    TEST_ASSERT(unified_cache_shutdown_state_clean(), "global cache retained owners after explicit shutdown");
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
    alloc_handle looked{};
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

int main(int argc, char ** argv) {
    fprintf(stderr, "===========================================\n");
    fprintf(stderr, "Unified Runtime Allocator Tests\n");
    fprintf(stderr, "===========================================\n");

    if (std::getenv("GGML_SYCL_PINNED_CHUNK_MB") == nullptr) {
        set_env_var("GGML_SYCL_PINNED_CHUNK_MB", "16");
    }

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

    // This child intentionally omits explicit shutdown. Its ordinary return
    // exercises the production fallback where g_device_caches destroys its
    // cache after main. Initializing the zone-sizing diagnostics after the
    // global cache registry reproduces the historical destructor-order crash.
    if (argc == 2 && std::strcmp(argv[1], "--static-destruction-child") == 0) {
        zone_sizing_record_observation("static-destruction-child");
        return get_unified_cache(q) ? 0 : 1;
    }

    bool ok = true;
    enable_strict_mode_env();
    ok &= reserve_allocate_success_registers_pointer(q);
    ok &= arena_registry_commit_failure_rolls_back(q);
    ok &= allocate_failure_rolls_back_budget(q);
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
    ok &= host_zone_reset_trims_released_offload_pool_slots(q);
    ok &= arena_owned_shutdown_and_lifecycle_serialization(q);
    ok &= arena_shutdown_drains_dma_and_bcs(q);
    ok &= exact_scratch_shutdown_record_survives_for_retry(q);
    ok &= scratch_regrow_refusal_preserves_exact_record(q);
    ok &= concurrent_settle_destroy_closing_wins(q);
    ok &= global_cache_static_destruction_exits_cleanly(argv[0]);
    // The fixture owns a process-global cache, so drain it while q and the SYCL
    // runtime are still alive rather than relying on static destruction.
    ok &= explicit_global_cache_shutdown_is_clean();

    fprintf(stderr, "-------------------------------------------\n");
    fprintf(stderr, "Tests: %d run, %d passed\n", g_tests_run, g_tests_passed);
    return ok ? 0 : 1;
}
