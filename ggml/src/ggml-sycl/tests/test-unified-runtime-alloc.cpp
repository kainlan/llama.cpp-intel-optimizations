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
#include "sycl-spin-kernel.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
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
    alloc_request req;
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
    alloc_metadata looked{};
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
    // Mirrors unified-cache.cpp's fragmentation path: the growth request must
    // guarantee that a fresh chunk can hold the whole allocation.
    const size_t need          = request + align;
    TEST_ASSERT(pool.grow_zone(host_zone_id::WEIGHT, need), "grow_zone should succeed within a 128 MiB budget");
    const size_t used_before = pool.zone_used(host_zone_id::WEIGHT);

    void * ptr = pool.zone_alloc(host_zone_id::WEIGHT, request, align);
    TEST_ASSERT(ptr != nullptr, "zone_alloc of 3 chunks after grow_zone should return a contiguous pointer");

    const size_t cap_after    = pool.zone_capacity(host_zone_id::WEIGHT);
    const size_t chunks_after = pool.chunk_count();
    const size_t aligned_need = (need + align - 1) & ~(align - 1);
    TEST_ASSERT(chunks_after == chunks_before + 1, "oversize growth should add exactly one chunk");
    TEST_ASSERT(cap_after - cap_before == aligned_need,
                "zone_capacity should grow by the actual bytes of the oversize chunk");
    // TLSF does not split off a tail smaller than its minimum block, so the
    // 64-byte slack rides along with the allocation: the accounted bytes are
    // at least the request and at most the new chunk.
    const size_t used_delta = pool.zone_used(host_zone_id::WEIGHT) - used_before;
    TEST_ASSERT(used_delta >= request && used_delta <= aligned_need,
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
    TEST_ASSERT(ggml_backend_sycl_stage_inventory_plan(&inventory, nullptr, true) == GGML_SYCL_LIFECYCLE_OK,
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
        plan.max_tensor_bytes + 32 * mib);
    const size_t host_zones = plan.host_zone_weight_bytes + std::max(plan.host_zone_kv_bytes, 64 * mib) +
                             plan.host_zone_staging_bytes + host_scratch;
    const size_t host_runtime = plan.onednn_scratchpad_bytes + plan.dma_staging_pool_bytes +
                                plan.pp_pipeline_scratch_bytes + plan.pp_moe_onednn_scratch_bytes;
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
    // Now cross the real late-inventory boundary. Its tiered_headroom
    // diagnostic (base_mem/4 when weights exceed shared capacity) is not a
    // second budget subtraction: g_tiered_headroom_reserve has no readers.
    TEST_ASSERT(ggml_backend_sycl_stage_inventory_plan(&inventory, nullptr, false) == GGML_SYCL_LIFECYCLE_OK,
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
// buffer's control kept the cache registered in g_device_caches, but a
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
// g_device_caches, but a subsequent onednn_graph_scratch_alloc() on that
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
    // exercises the production fallback where g_device_caches destroys its
    // cache after main. Initializing the zone-sizing diagnostics after the
    // global cache registry reproduces the historical destructor-order crash.
    if (argc == 2 && std::strcmp(argv[1], "--static-destruction-child") == 0) {
        zone_sizing_record_observation("static-destruction-child");
        return get_unified_cache(q) ? 0 : 1;
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
    ok &= host_zone_reset_trims_released_offload_pool_slots(q);
    ok &= independent_exact_token_defers_owned_release(q);
    ok &= arena_owned_shutdown_and_lifecycle_serialization(q);
    ok &= arena_shutdown_drains_dma_and_bcs(q);
    ok &= exact_scratch_shutdown_record_survives_for_retry(q);
    ok &= scratch_regrow_refusal_preserves_exact_record(q);
    ok &= concurrent_settle_destroy_closing_wins(q);
    ok &= accepted_control_publication_is_atomic_with_close_snapshot();
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
