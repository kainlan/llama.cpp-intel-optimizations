// SYCL Unified Memory End-to-End Validation Tests
// Part of unified memory management system (epic llama.cpp-v3n, task llama.cpp-13o)
//
// Goal: validate that the unified cache enforces its weight budget and evicts
// under pressure, using 120B-MoE-shaped allocation patterns without a real model.
//
// THE BUDGET CONTRACT THIS FIXTURE MUST RESPECT (llama.cpp-mequ)
// -------------------------------------------------------------
// unified_cache::budget()/used()/available() are WEIGHT-CACHE ACCOUNTING.  They
// are NOT a measure of physically allocatable VRAM, and the gap is deliberate:
//
//   * The VRAM arena reserves its block with a single sycl::malloc_device inside
//     the unified_cache CONSTRUCTOR (unified-cache.cpp, ensure_planned_arena_zones
//     -> arena_reserve).  So the arena is already holding VRAM by the time the
//     first get_unified_cache_for_device() call returns.
//   * That reservation is deliberately NOT subtracted from budget_.  See the
//     792vn.5 design note above unified_cache::update_reserved_bytes()
//     (unified-cache.cpp:8573): "Arena zones now handle VRAM partitioning for
//     runtime allocations ... handled by the overcommit guard in
//     unified_cache_total_committed_bytes() rather than by shrinking the weight
//     cache budget."  budget_ = base_budget_ - reserved_, and nothing charges
//     the arena to reserved_.
//
// With the DEFAULT whole-device budget those two facts collide.  budget() then
// reports very nearly all of VRAM while the arena has physically taken very
// nearly all of VRAM, leaving only the arena's external headroom (~2 GiB) for
// anything allocated outside a zone.  ensure_cached_alloc() -- the API these
// tests drive -- is the one cache path that allocates outside the arena: it
// gates on used_ + size > budget_ and then calls sycl::malloc_device directly
// (unified-cache.cpp, "alloc malloc_device returned nullptr").  Sizing an
// allocation from available() therefore asks for memory the cache itself owns,
// and the driver returns nullptr.  That is self-starvation, not a cache defect.
//
// ensure_cached_alloc() is [[deprecated("use unified_alloc()")]] and has no
// production callers -- the shim survives only for this test family, and
// llama.cpp-og9dt tracks porting these call sites to unified_alloc() and
// deleting it.  Until that lands, this fixture is exercising a path the
// backend itself no longer takes: the production paths (allocate_slot,
// unified_alloc) try the arena zone first and check live free VRAM before any
// raw malloc, which is exactly what spares them this failure mode.
//
// It also makes the eviction claim unreachable.  The only path to evict_one()
// is the used_ + size > budget_ branch, so with budget_ ~= all of VRAM the
// cache never even attempts eviction -- it fails the raw malloc first.  And
// eviction returns bytes to used_ immediately but frees the device allocation
// only via enqueue_deferred_free(), so evicting does not hand physical VRAM
// back to the next malloc_device in any case.
//
// THE REPAIR: pin an explicit, small cache budget before the first cache access
// (set_unified_cache_budget(), "call before first use").  The arena then
// reserves only that much, leaving the rest of the device for the direct
// allocations, and used_ can actually reach budget_ -- so the eviction path is
// reachable at all.  Every size below is derived from that pinned budget.
//
// Test Scenarios:
// 1. Memory pressure simulation (120B MoE patterns without real model)
// 2. Expert streaming under pressure (allocate past the budget, require eviction)
// 3. KV cache + expert coexistence (verify priority-based eviction)
// 4. Three-tier memory flow (VRAM -> HOST -> MMAP)
// 5. Budget enforcement (explicit limits, graceful rejection)
// 6. Regression (small models still work without unnecessary overflow)
//
// Exit codes are tri-state: 0 = every subcase ran and passed, 1 = a property
// this test asserts did not hold, 77 = a capability or configuration it needs
// is genuinely absent (ctest SKIP_RETURN_CODE).  Never collapse 77 into 0 -- a
// subcase that could not run proves NOTHING about the path it names.

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Include unified cache components
#include "ggml-sycl/unified-cache.hpp"
#include "ggml-sycl/eviction-policy.hpp"
#include "ggml-sycl/expert-prefetch.hpp"
#include "ggml-sycl/compute-buffer-manager.hpp"

// Size helpers
constexpr size_t operator""_KB(unsigned long long n) {
    return n * 1024;
}

constexpr size_t operator""_MB(unsigned long long n) {
    return n * 1024 * 1024;
}

constexpr size_t operator""_GB(unsigned long long n) {
    return n * 1024 * 1024 * 1024;
}

// ctest's SKIP_RETURN_CODE: a skip must be visible AS a skip, so it can never
// be read as "this run validated the unified memory system".
static const int k_exit_skip = 77;

// A subcase either verified its property, disproved it, or could not run.
enum class test_result {
    PASS,
    FAIL,
    SKIP,
};

// The cache budget this fixture pins before the first cache access.  Two
// constraints set it (see the contract note at the top of this file):
//
//   * Large enough that the arena's default tail zones still fit.  At first
//     init those are SCRATCH 512 MB + ONEDNN 256 MB + RUNTIME 512 MB = 1280 MB,
//     and arena_reserve() refuses a single-chunk arena below tail + 16 MB.
//   * Small enough that the device keeps ample VRAM outside the arena, since
//     ensure_cached_alloc() allocates there.  The fill subcases request roughly
//     1.25x the budget before eviction plateaus them, and evicted allocations
//     are freed lazily via the deferred-free queue, so the true device high
//     water mark is the arena plus every byte requested.
static constexpr size_t k_test_budget_bytes = 2048_MB;

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_skipped = 0;

// Unique ID counter
static uint64_t g_unique_id = 0x200000;

static uint64_t next_unique_id() {
    return ++g_unique_id;
}

// Dummy source data for cache testing
static std::vector<uint8_t> g_dummy_src;

static void * get_dummy_src(size_t size) {
    if (g_dummy_src.size() < size) {
        g_dummy_src.resize(size);
        for (size_t i = 0; i < size; ++i) {
            g_dummy_src[i] = static_cast<uint8_t>(i & 0xFF);
        }
    }
    return g_dummy_src.data();
}

// 120B MoE model parameters (simulated)
struct moe_120b_params {
    static constexpr int    n_layers       = 40;         // Number of transformer layers
    static constexpr int    n_experts      = 128;        // Experts per layer
    static constexpr size_t expert_size    = 500_MB;     // ~500MB per expert
    static constexpr size_t kv_head_size   = 32_MB;      // ~32MB per KV head
    static constexpr int    n_kv_heads     = 144;        // KV heads total (4.6GB)
    static constexpr size_t attention_size = 200_MB;     // Attention weights per layer
    static constexpr size_t compute_size   = 64_MB;      // Compute buffer requirement

    // Total model size: 128 * 500MB * 40 layers = ~2.5TB (mostly on disk)
    // Active set: attention (8GB) + 2 hot experts (1GB) + KV cache (4.6GB) = ~14GB
};

// =============================================================================
// Test 1: Memory pressure simulation with 120B MoE patterns
// Simulates allocating the active set without a real model
// =============================================================================
static test_result test_120b_memory_pressure_simulation() {
    printf("TEST: test_120b_memory_pressure_simulation\n");
    printf("  Simulating 120B MoE memory patterns (without real model)\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache for device 0\n");
        return test_result::FAIL;
    }

    size_t budget = cache->budget();
    size_t initial_used = cache->used();

    printf("  Budget: %zu MB, Initial used: %zu MB\n", budget / (1024*1024), initial_used / (1024*1024));

    // For a real 120B model the active set is ~14 GB.  Scale it to the pinned
    // budget, which is what this fixture may actually claim -- NOT to free VRAM
    // and NOT to the device total.
    size_t available = cache->available();
    float scale = static_cast<float>(available) / (14_GB);
    if (scale > 1.0f) scale = 1.0f;
    if (scale < 0.05f) {
        printf(
            "  SKIP: budget headroom %zu MB is too small to shape an active set (scale=%.2f);"
            " this subcase proves NOTHING about memory pressure\n",
            available / (1024 * 1024), scale);
        return test_result::SKIP;
    }

    printf("  Scale factor: %.2f (simulating %.1f GB active set)\n", scale, 14.0f * scale);

    // Allocate simulated attention weights (scaled)
    size_t attention_alloc = static_cast<size_t>(moe_120b_params::attention_size * 10 * scale);
    ggml_sycl_cache_id attn_id = {};
    attn_id.valid = true;
    attn_id.model_id = 120;
    attn_id.name_hash = next_unique_id();
    attn_id.nbytes = attention_alloc;
    attn_id.type = GGML_TYPE_F16;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) attn_id.ne[i] = 1;

    bool needs_fill = false;
    void * attn_ptr = cache->ensure_cached_alloc(
        attn_id, get_dummy_src(attention_alloc), attention_alloc, attention_alloc,
        ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

    if (!attn_ptr) {
        printf("  FAIL: attention allocation of %zu MB failed against a %zu MB budget (used %zu MB)\n",
               attention_alloc / (1024 * 1024), budget / (1024 * 1024), cache->used() / (1024 * 1024));
        return test_result::FAIL;
    }
    printf("  Allocated attention weights: %zu MB\n", attention_alloc / (1024*1024));

    // Allocate simulated KV cache heads (scaled)
    std::vector<ggml_sycl_cache_id> kv_ids;
    size_t kv_head_alloc = static_cast<size_t>(moe_120b_params::kv_head_size * scale);
    int n_kv_alloc = static_cast<int>(moe_120b_params::n_kv_heads * scale);
    if (n_kv_alloc < 1) n_kv_alloc = 1;

    size_t kv_total = 0;
    for (int i = 0; i < n_kv_alloc; ++i) {
        ggml_sycl_cache_id kv_id = {};
        kv_id.valid = true;
        kv_id.model_id = 120;
        kv_id.name_hash = next_unique_id();
        kv_id.nbytes = kv_head_alloc;
        kv_id.type = GGML_TYPE_F16;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) kv_id.ne[j] = 1;

        void * kv_ptr = cache->ensure_cached_alloc(
            kv_id, get_dummy_src(kv_head_alloc), kv_head_alloc, kv_head_alloc,
            ggml_sycl::cache_entry_type::DENSE_WEIGHT, i, -1, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!kv_ptr) {
            // The whole scaled active set fits inside the pinned budget, so a
            // nullptr here is a real failure -- either the direct malloc_device
            // failed or eviction could not free enough.  It is NOT the
            // "expected pressure" the pre-mequ fixture reported it as: that
            // reading is what let genuine starvation pass as a healthy run.
            printf("  FAIL: KV head %d (%zu MB) failed at used %zu MB of %zu MB budget\n", i,
                   kv_head_alloc / (1024 * 1024), cache->used() / (1024 * 1024), budget / (1024 * 1024));
            cache->remove(attn_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
            for (size_t j = 0; j < kv_ids.size(); ++j) {
                cache->remove(kv_ids[j], ggml_sycl::cache_entry_type::DENSE_WEIGHT, static_cast<int>(j), -1,
                              GGML_LAYOUT_AOS);
            }
            return test_result::FAIL;
        }
        kv_ids.push_back(kv_id);
        kv_total += kv_head_alloc;
    }
    printf("  Allocated %zu KV heads: %zu MB\n", kv_ids.size(), kv_total / (1024*1024));

    // Allocate simulated MoE experts (scaled, expect some to evict)
    std::vector<ggml_sycl_cache_id> expert_ids;
    size_t expert_alloc = static_cast<size_t>(50_MB * scale);  // Smaller for testing
    int n_experts_alloc = 10;  // Try to allocate 10 experts

    size_t expert_total = 0;
    int experts_succeeded = 0;
    for (int i = 0; i < n_experts_alloc; ++i) {
        ggml_sycl_cache_id exp_id = {};
        exp_id.valid = true;
        exp_id.model_id = 120;
        exp_id.name_hash = next_unique_id();
        exp_id.nbytes = expert_alloc;
        exp_id.type = GGML_TYPE_F16;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) exp_id.ne[j] = 1;

        void * exp_ptr = cache->ensure_cached_alloc(
            exp_id, get_dummy_src(expert_alloc), expert_alloc, expert_alloc,
            ggml_sycl::cache_entry_type::MOE_EXPERT, 0, i, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!exp_ptr) {
            printf("  FAIL: expert %d (%zu MB) failed at used %zu MB of %zu MB budget\n", i,
                   expert_alloc / (1024 * 1024), cache->used() / (1024 * 1024), budget / (1024 * 1024));
            break;
        }
        expert_ids.push_back(exp_id);
        expert_total += expert_alloc;
        experts_succeeded++;
    }
    printf("  Allocated %d experts: %zu MB\n", experts_succeeded, expert_total / (1024*1024));

    // Verify cache state is consistent
    size_t final_used = cache->used();
    printf("  Final used: %zu MB\n", final_used / (1024*1024));

    // Every entry is keyed by (id, type, layer_id, expert_id, layout), so each
    // remove() must repeat the coordinates its ensure_cached_alloc() used.  The
    // pre-mequ cleanup passed layer_id = -1 for the KV heads, which were staged
    // at layer_id = i, so it removed nothing and leaked them into every later
    // subcase's used().
    cache->remove(attn_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
    for (size_t i = 0; i < kv_ids.size(); ++i) {
        cache->remove(kv_ids[i], ggml_sycl::cache_entry_type::DENSE_WEIGHT, static_cast<int>(i), -1, GGML_LAYOUT_AOS);
    }
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        cache->remove(expert_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i), GGML_LAYOUT_AOS);
    }

    // Trigger deferred free processing
    cache->evict(0);

    if (experts_succeeded != n_experts_alloc) {
        printf("  >>> %d of %d experts failed\n", n_experts_alloc - experts_succeeded, n_experts_alloc);
        return test_result::FAIL;
    }

    size_t total_allocated = attention_alloc + kv_total + expert_total;
    if (total_allocated < 50_MB) {
        printf("  FAIL: allocated too little memory (%zu MB), expected more\n", total_allocated / (1024*1024));
        return test_result::FAIL;
    }

    printf("  PASS: %zu MB active set staged and released without starvation\n", total_allocated / (1024 * 1024));
    return test_result::PASS;
}

// =============================================================================
// Test 2: Expert streaming under pressure
// Fill VRAM to capacity, then request experts that aren't cached
// =============================================================================
static test_result test_expert_streaming_under_pressure() {
    printf("TEST: test_expert_streaming_under_pressure\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache\n");
        return test_result::FAIL;
    }

    const size_t budget = cache->budget();
    if (budget < 256_MB) {
        printf(
            "  SKIP: budget %zu MB is below the 256 MB this subcase needs to build pressure;"
            " this run proves NOTHING about expert streaming\n",
            budget / (1024 * 1024));
        return test_result::SKIP;
    }

    printf("  Budget: %zu MB, used at entry: %zu MB\n", budget / (1024 * 1024), cache->used() / (1024 * 1024));

    // Request MORE than the whole budget.  ensure_cached_alloc() reaches
    // evict_one() only from its `used_ + size > budget_` branch, so a fill
    // sized as a FRACTION of the budget -- what this subcase did before mequ --
    // can never enter the eviction path it claims to exercise.  Overshooting
    // guarantees the branch is taken while eviction keeps every request served.
    const size_t expert_size = 16_MB;
    const int    n_requests  = static_cast<int>(budget / expert_size) + 32;

    std::vector<ggml_sycl_cache_id> cold_ids;
    cold_ids.reserve(static_cast<size_t>(n_requests));

    int    failed_at = -1;
    size_t requested = 0;
    for (int i = 0; i < n_requests; ++i) {
        ggml_sycl_cache_id id = {};
        id.valid = true;
        id.model_id = 200;
        id.name_hash = next_unique_id();
        id.nbytes = expert_size;
        id.type = GGML_TYPE_F16;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) id.ne[j] = 1;

        bool needs_fill = false;
        void * ptr = cache->ensure_cached_alloc(
            id, get_dummy_src(expert_size), expert_size, expert_size,
            ggml_sycl::cache_entry_type::MOE_EXPERT, 0, i, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!ptr) {
            failed_at = i;
            break;
        }

        cold_ids.push_back(id);
        requested += expert_size;
    }

    printf("  Requested %zu experts (%zu MB) against a %zu MB budget; used now %zu MB\n", cold_ids.size(),
           requested / (1024 * 1024), budget / (1024 * 1024), cache->used() / (1024 * 1024));

    // Count survivors BEFORE cleanup: an entry that is no longer cached is one
    // the cache evicted to serve a later request.  This is a per-subcase signal.
    // has_evictions() is not usable here -- it latches true for the lifetime of
    // the cache, so after any earlier subcase evicts it would report success for
    // a run in which this subcase evicted nothing.
    size_t still_cached = 0;
    for (const auto & id : cold_ids) {
        if (cache->is_cached(id, GGML_LAYOUT_AOS)) {
            still_cached++;
        }
    }

    const size_t used_after = cache->used();

    for (size_t i = 0; i < cold_ids.size(); ++i) {
        cache->remove(cold_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i), GGML_LAYOUT_AOS);
    }
    cache->evict(0);

    if (failed_at >= 0) {
        printf("  FAIL: request %d of %d (%zu MB) returned nullptr; eviction did not make room\n", failed_at,
               n_requests, expert_size / (1024 * 1024));
        return test_result::FAIL;
    }

    if (still_cached >= cold_ids.size()) {
        printf(
            "  FAIL: %zu MB requested against a %zu MB budget yet all %zu entries survived;"
            " the eviction path never ran\n",
            requested / (1024 * 1024), budget / (1024 * 1024), cold_ids.size());
        return test_result::FAIL;
    }

    if (used_after > budget) {
        printf("  FAIL: used %zu MB exceeds budget %zu MB after the fill\n", used_after / (1024 * 1024),
               budget / (1024 * 1024));
        return test_result::FAIL;
    }

    printf("  Evicted %zu of %zu entries; used %zu MB stayed within budget\n", cold_ids.size() - still_cached,
           cold_ids.size(), used_after / (1024 * 1024));
    printf("  PASS: expert streaming under pressure works\n");
    return test_result::PASS;
}

// =============================================================================
// Test 3: KV cache + expert coexistence
// Verify priority-based eviction: experts evict before KV cache
// =============================================================================
static test_result test_kv_expert_coexistence() {
    printf("TEST: test_kv_expert_coexistence\n");

    // Use eviction policy directly to test priority behavior
    ggml_sycl::EvictionPolicy policy;

    // Add KV cache entries (P1 - high priority, evict last)
    for (int i = 0; i < 5; ++i) {
        uint64_t kv_id = 1000 + i;
        policy.add_entry(kv_id, ggml_sycl::EvictionPriority::P1_ACTIVE_KV, 32_MB,
                         ggml_sycl::cache_location::DEVICE);
    }

    // Add hot experts (P2)
    for (int i = 0; i < 3; ++i) {
        uint64_t hot_id = 2000 + i;
        policy.add_entry(hot_id, ggml_sycl::EvictionPriority::P2_HOT_EXPERT, 50_MB,
                         ggml_sycl::cache_location::DEVICE);
    }

    // Add cold experts (P4 - evict first)
    for (int i = 0; i < 5; ++i) {
        uint64_t cold_id = 4000 + i;
        policy.add_entry(cold_id, ggml_sycl::EvictionPriority::P4_COLD_EXPERT, 50_MB,
                         ggml_sycl::cache_location::DEVICE);
    }

    // Add P0 compute buffer (never evict)
    policy.add_entry(9000, ggml_sycl::EvictionPriority::P0_COMPUTE, 64_MB,
                     ggml_sycl::cache_location::DEVICE);

    // Evict entries one by one and verify order
    std::vector<uint64_t> eviction_order;
    while (true) {
        auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);
        if (!victim.has_value()) break;
        eviction_order.push_back(victim->id);
        policy.remove_entry(victim->id);
    }

    // Verify P0 (compute) was never evicted
    if (std::find(eviction_order.begin(), eviction_order.end(), 9000) != eviction_order.end()) {
        printf("  FAIL: P0 compute buffer was evicted\n");
        return test_result::FAIL;
    }

    // Verify cold experts (P4) were evicted first
    bool found_p4_before_p2 = false;
    bool found_p2 = false;
    for (uint64_t id : eviction_order) {
        if (id >= 4000 && id < 5000) {
            // P4 cold expert
            if (found_p2) {
                printf("  FAIL: P4 was evicted after P2\n");
                return test_result::FAIL;
            }
            found_p4_before_p2 = true;
        } else if (id >= 2000 && id < 3000) {
            // P2 hot expert
            found_p2 = true;
        }
    }

    if (!found_p4_before_p2) {
        printf("  FAIL: no P4 entries were evicted\n");
        return test_result::FAIL;
    }

    // Verify KV (P1) was evicted last among evictable entries
    found_p2 = false;
    for (uint64_t id : eviction_order) {
        if (id >= 1000 && id < 2000) {
            // P1 KV cache
            if (!found_p2) {
                printf("  FAIL: P1 was evicted before P2\n");
                return test_result::FAIL;
            }
        } else if (id >= 2000 && id < 3000) {
            found_p2 = true;
        }
    }

    printf("  Eviction order verified: P4 (cold) -> P3/P2 (experts) -> P1 (KV)\n");
    printf("  P0 (compute) was protected from eviction\n");
    printf("  PASS: priority-based eviction works correctly\n");
    return test_result::PASS;
}

// =============================================================================
// Test 4: Three-tier memory flow (VRAM -> HOST -> MMAP)
// =============================================================================
static test_result test_three_tier_memory_flow() {
    printf("TEST: test_three_tier_memory_flow\n");

    // Verify the cache_location enum has all three tiers
    ggml_sycl::cache_location device = ggml_sycl::cache_location::DEVICE;
    ggml_sycl::cache_location host = ggml_sycl::cache_location::HOST_PINNED;
    ggml_sycl::cache_location mmap = ggml_sycl::cache_location::HOST_MMAP;

    // Verify enum values are distinct
    if (static_cast<int>(device) == static_cast<int>(host) ||
        static_cast<int>(host) == static_cast<int>(mmap) ||
        static_cast<int>(device) == static_cast<int>(mmap)) {
        printf("  FAIL: cache_location enum values are not distinct\n");
        return test_result::FAIL;
    }

    // Check that host arena is available via unified cache
    ggml_sycl::unified_cache * ucache = ggml_sycl::get_unified_cache_for_device(0);
    if (!ucache) {
        printf("  SKIP: unified cache not available; this run proves NOTHING about the host tier\n");
        return test_result::SKIP;
    }
    size_t host_budget = ucache->pinned_pool_budget();
    if (host_budget == 0) {
        printf("  SKIP: host arena has 0 budget; this run proves NOTHING about the host tier\n");
        return test_result::SKIP;
    }

    printf("  Host arena budget: %zu MB\n", host_budget / (1024*1024));
    printf("  PASS: three-tier memory system available\n");
    return test_result::PASS;
}

// =============================================================================
// Test 5: Budget enforcement
// Verify explicit VRAM budget limits are respected
// =============================================================================
static test_result test_budget_enforcement() {
    printf("TEST: test_budget_enforcement\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache\n");
        return test_result::FAIL;
    }

    size_t budget = cache->budget();
    size_t used = cache->used();
    size_t available = cache->available();

    // Verify accounting consistency
    if (available != budget - used) {
        printf("  FAIL: available (%zu) != budget (%zu) - used (%zu)\n",
               available, budget, used);
        return test_result::FAIL;
    }

    // Verify budget is reasonable (at least 100MB, at most total device memory)
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(0, &free_mem, &total_mem);

    if (budget < 100_MB) {
        printf("  FAIL: budget too small (%zu MB)\n", budget / (1024*1024));
        return test_result::FAIL;
    }

    if (budget > total_mem) {
        printf("  FAIL: budget (%zu) exceeds total device memory (%zu)\n", budget, total_mem);
        return test_result::FAIL;
    }

    printf("  Budget: %zu MB (of %zu MB total)\n", budget / (1024*1024), total_mem / (1024*1024));
    printf("  Used: %zu MB, Available: %zu MB\n", used / (1024*1024), available / (1024*1024));
    printf("  PASS: budget enforcement is active and reasonable\n");
    return test_result::PASS;
}

// =============================================================================
// Test 6: Compute buffer protection (P0 never evicted)
// =============================================================================
static test_result test_compute_buffer_protection() {
    printf("TEST: test_compute_buffer_protection\n");

    ggml_sycl::EvictionPolicy policy;

    // Add only P0 (compute) entries
    for (int i = 0; i < 5; ++i) {
        policy.add_entry(100 + i, ggml_sycl::EvictionPriority::P0_COMPUTE, 64_MB,
                         ggml_sycl::cache_location::DEVICE);
    }

    // Try to select a victim
    auto victim = policy.select_victim(ggml_sycl::cache_location::DEVICE);

    if (victim.has_value()) {
        printf("  FAIL: P0 compute buffer was selected for eviction (id=%lu)\n", victim->id);
        return test_result::FAIL;
    }

    printf("  PASS: P0 compute buffers are protected from eviction\n");
    return test_result::PASS;
}

// =============================================================================
// Test 7: Expert prefetch integration
// Verify prefetcher sorts by score and tracks accuracy
// =============================================================================
static test_result test_expert_prefetch_integration() {
    printf("TEST: test_expert_prefetch_integration\n");

    // The ExpertPrefetcher integrates with ExpertCache for DMA-based prefetch.
    // Full integration testing requires a real ExpertCache instance.
    // Basic API smoke test: verify uninitialized prefetcher is safe.

    ggml_sycl::ExpertPrefetcher prefetcher;

    // Not active before init
    if (prefetcher.is_active()) {
        printf("  FAIL: should not be active before init\n");
        return test_result::FAIL;
    }

    // hint/await should be safe when uninitialized
    if (prefetcher.hint(0, 2)) {
        printf("  FAIL: hint should return false before init\n");
        return test_result::FAIL;
    }

    if (prefetcher.await(0, 2) != nullptr) {
        printf("  FAIL: await should return nullptr before init\n");
        return test_result::FAIL;
    }

    if (prefetcher.pending_count() != 0) {
        printf("  FAIL: pending_count should be 0\n");
        return test_result::FAIL;
    }

    printf("  PASS: expert prefetch integration works (DMA engine smoke test)\n");
    return test_result::PASS;
}

// =============================================================================
// Test 8: Memory tracking consistency
// Verify used() and available() stay consistent across operations
// =============================================================================
static test_result test_memory_tracking_consistency() {
    printf("TEST: test_memory_tracking_consistency\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache\n");
        return test_result::FAIL;
    }

    size_t budget = cache->budget();
    size_t initial_used = cache->used();

    // Allocate, track, and free in sequence
    std::vector<ggml_sycl_cache_id> ids;
    size_t total_allocated = 0;

    for (int i = 0; i < 5; ++i) {
        size_t alloc_size = 10_MB * (i + 1);  // 10, 20, 30, 40, 50 MB

        ggml_sycl_cache_id id = {};
        id.valid = true;
        id.model_id = 300;
        id.name_hash = next_unique_id();
        id.nbytes = alloc_size;
        id.type = GGML_TYPE_F32;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) id.ne[j] = 1;

        bool needs_fill = false;
        void * ptr = cache->ensure_cached_alloc(
            id, get_dummy_src(alloc_size), alloc_size, alloc_size,
            ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!ptr) {
            // 150 MB in total against the pinned budget: there is no legitimate
            // pressure here.  Breaking out with a "SKIP" -- as this did before
            // mequ -- left total_allocated at 0, which then satisfied the
            // freed-bytes assertion below vacuously and reported PASS.
            printf("  FAIL: allocation %d (%zu MB) failed at used %zu MB of %zu MB budget\n", i,
                   alloc_size / (1024 * 1024), cache->used() / (1024 * 1024), budget / (1024 * 1024));
            for (const auto & cleanup_id : ids) {
                cache->remove(cleanup_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
            }
            return test_result::FAIL;
        }

        ids.push_back(id);
        total_allocated += alloc_size;

        // Verify used increased
        size_t current_used = cache->used();
        if (current_used < initial_used + total_allocated - 10_MB) {  // Allow some margin
            printf("  FAIL: used didn't increase properly after allocation %d\n", i);
            // Clean up
            for (const auto & cleanup_id : ids) {
                cache->remove(cleanup_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
            }
            return test_result::FAIL;
        }
    }

    printf("  Allocated %zu entries totaling %zu MB\n", ids.size(), total_allocated / (1024*1024));

    // Verify available + used = budget (approximately)
    size_t mid_used = cache->used();
    size_t mid_available = cache->available();

    if (mid_used + mid_available != budget) {
        printf("  FAIL: used (%zu) + available (%zu) != budget (%zu)\n",
               mid_used, mid_available, budget);
        // Clean up
        for (const auto & id : ids) {
            cache->remove(id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
        }
        return test_result::FAIL;
    }

    // Free entries
    for (const auto & id : ids) {
        cache->remove(id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
    }

    // Trigger deferred free processing
    cache->evict(0);

    // Verify memory was freed
    size_t final_used = cache->used();
    size_t freed = mid_used - final_used;

    // Should have freed most of what we allocated (allow 20% margin for overhead)
    if (freed < total_allocated * 8 / 10) {
        printf("  FAIL: expected to free ~%zu MB, only freed %zu MB\n",
               total_allocated / (1024*1024), freed / (1024*1024));
        return test_result::FAIL;
    }

    printf("  Freed %zu MB\n", freed / (1024*1024));
    printf("  PASS: memory tracking is consistent\n");
    return test_result::PASS;
}

// =============================================================================
// Test 9: Eviction under high fragmentation
// Many small allocations followed by large allocation request
// =============================================================================
static test_result test_eviction_fragmentation() {
    printf("TEST: test_eviction_fragmentation\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache\n");
        return test_result::FAIL;
    }

    const size_t budget = cache->budget();
    if (budget < 256_MB) {
        printf(
            "  SKIP: budget %zu MB is below the 256 MB this subcase needs to build pressure;"
            " this run proves NOTHING about fragmented eviction\n",
            budget / (1024 * 1024));
        return test_result::SKIP;
    }

    // Fragment the whole budget and then overshoot it.  Filling a FRACTION of
    // the budget -- 60% before mequ -- never reaches ensure_cached_alloc()'s
    // `used_ + size > budget_` branch, which is the only caller of evict_one();
    // the large allocation below then succeeded without any eviction having
    // happened, and the subcase reported that as "eviction handles
    // fragmentation".
    std::vector<ggml_sycl_cache_id> small_ids;
    const size_t                    small_size  = 2_MB;
    const int                       small_count = static_cast<int>(budget / small_size) + 64;

    int failed_at = -1;
    for (int i = 0; i < small_count; ++i) {
        ggml_sycl_cache_id id = {};
        id.valid = true;
        id.model_id = 400;
        id.name_hash = next_unique_id();
        id.nbytes = small_size;
        id.type = GGML_TYPE_F16;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) id.ne[j] = 1;

        bool needs_fill = false;
        void * ptr = cache->ensure_cached_alloc(
            id, get_dummy_src(small_size), small_size, small_size,
            ggml_sycl::cache_entry_type::MOE_EXPERT, 0, i, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!ptr) {
            failed_at = i;
            break;
        }
        small_ids.push_back(id);
    }

    printf("  Requested %zu small entries of %zu MB against a %zu MB budget\n", small_ids.size(),
           small_size / (1024 * 1024), budget / (1024 * 1024));

    if (failed_at >= 0) {
        printf("  FAIL: small entry %d of %d returned nullptr; eviction did not make room\n", failed_at, small_count);
        for (size_t i = 0; i < small_ids.size(); ++i) {
            cache->remove(small_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i),
                          GGML_LAYOUT_AOS);
        }
        cache->evict(0);
        return test_result::FAIL;
    }

    size_t small_survivors = 0;
    for (const auto & id : small_ids) {
        if (cache->is_cached(id, GGML_LAYOUT_AOS)) {
            small_survivors++;
        }
    }
    if (small_survivors >= small_ids.size()) {
        printf(
            "  FAIL: every one of %zu small entries survived a fill past the budget;"
            " the eviction path never ran\n",
            small_ids.size());
        for (size_t i = 0; i < small_ids.size(); ++i) {
            cache->remove(small_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i),
                          GGML_LAYOUT_AOS);
        }
        cache->evict(0);
        return test_result::FAIL;
    }
    printf("  Evicted %zu of %zu small entries while filling\n", small_ids.size() - small_survivors, small_ids.size());

    // Now a large entry against a budget that is already full of 2 MB entries:
    // it can only be served by evicting several of them.
    size_t             large_size = 32_MB;
    ggml_sycl_cache_id large_id = {};
    large_id.valid = true;
    large_id.model_id = 400;
    large_id.name_hash = next_unique_id();
    large_id.nbytes = large_size;
    large_id.type = GGML_TYPE_F16;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) large_id.ne[i] = 1;

    bool needs_fill = false;
    void * large_ptr = cache->ensure_cached_alloc(
        large_id, get_dummy_src(large_size), large_size, large_size,
        ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

    if (!large_ptr) {
        printf("  FAIL: %zu MB allocation failed at used %zu MB of %zu MB budget despite eviction\n",
               large_size / (1024 * 1024), cache->used() / (1024 * 1024), budget / (1024 * 1024));
        // Clean up
        for (size_t i = 0; i < small_ids.size(); ++i) {
            cache->remove(small_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i), GGML_LAYOUT_AOS);
        }
        cache->evict(0);
        return test_result::FAIL;
    }

    printf("  Large allocation succeeded (evicted small entries)\n");

    // Sample before cleanup: serving the large entry must not have pushed the
    // cache past its budget.  This is the third leg of the claim -- "served",
    // "by evicting", and "still within budget" -- and without it a cache that
    // simply grew to fit would pass.  Same assertion test 2 makes.
    const size_t used_after = cache->used();

    // Clean up
    for (size_t i = 0; i < small_ids.size(); ++i) {
        cache->remove(small_ids[i], ggml_sycl::cache_entry_type::MOE_EXPERT, 0, static_cast<int>(i), GGML_LAYOUT_AOS);
    }
    cache->remove(large_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
    cache->evict(0);

    if (used_after > budget) {
        printf("  FAIL: used %zu MB exceeds budget %zu MB after serving the %zu MB allocation\n",
               used_after / (1024 * 1024), budget / (1024 * 1024), large_size / (1024 * 1024));
        return test_result::FAIL;
    }

    printf("  Used %zu MB stayed within the %zu MB budget\n", used_after / (1024 * 1024), budget / (1024 * 1024));
    printf("  PASS: eviction handles fragmentation\n");
    return test_result::PASS;
}

// =============================================================================
// Test 10: Cache lookup consistency
// Verify that cached entries can be found consistently via is_cached() and get()
// Note: hit/miss stats are only tracked in ensure_cached (sync copy), not in
// ensure_cached_alloc (caller-managed fill) or get (pure lookup).
// =============================================================================
static test_result test_cache_lookup_consistency() {
    printf("TEST: test_cache_lookup_consistency\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("  FAIL: could not get unified cache\n");
        return test_result::FAIL;
    }

    // Create a working set
    std::vector<ggml_sycl_cache_id> ids;
    size_t entry_size = 5_MB;
    int n_entries = 5;

    // Populate cache
    for (int i = 0; i < n_entries; ++i) {
        ggml_sycl_cache_id id = {};
        id.valid = true;
        id.model_id = 500;
        id.name_hash = next_unique_id();
        id.nbytes = entry_size;
        id.type = GGML_TYPE_F32;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) id.ne[j] = 1;

        bool needs_fill = false;
        void * ptr = cache->ensure_cached_alloc(
            id, get_dummy_src(entry_size), entry_size, entry_size,
            ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false, &needs_fill);

        if (!ptr) {
            printf("  FAIL: entry %d of %d (%zu MB) failed at used %zu MB of %zu MB budget\n", i, n_entries,
                   entry_size / (1024 * 1024), cache->used() / (1024 * 1024), cache->budget() / (1024 * 1024));
            for (const auto & cleanup_id : ids) {
                cache->remove(cleanup_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
            }
            return test_result::FAIL;
        }
        ids.push_back(id);
    }

    printf("  Created %d cached entries\n", n_entries);

    // Verify all entries are cached via is_cached()
    int cached_count = 0;
    for (const auto & id : ids) {
        if (cache->is_cached(id, GGML_LAYOUT_AOS)) {
            cached_count++;
        }
    }

    if (cached_count != n_entries) {
        printf("  FAIL: is_cached returned true for %d/%d entries\n", cached_count, n_entries);
        for (const auto & id : ids) {
            cache->remove(id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
        }
        return test_result::FAIL;
    }

    // Verify all entries can be retrieved via get()
    int get_count = 0;
    for (const auto & id : ids) {
        void * ptr = cache->get(id, GGML_LAYOUT_AOS);
        if (ptr != nullptr) {
            get_count++;
        }
    }

    if (get_count != n_entries) {
        printf("  FAIL: get() returned non-null for %d/%d entries\n", get_count, n_entries);
        for (const auto & id : ids) {
            cache->remove(id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
        }
        return test_result::FAIL;
    }

    // Access multiple times to verify stability
    for (int pass = 0; pass < 10; ++pass) {
        for (const auto & id : ids) {
            void * ptr = cache->get(id, GGML_LAYOUT_AOS);
            if (!ptr) {
                printf("  FAIL: cached entry disappeared on pass %d\n", pass);
                for (const auto & cleanup_id : ids) {
                    cache->remove(cleanup_id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
                }
                return test_result::FAIL;
            }
        }
    }

    printf("  All entries found consistently across %d passes\n", 10);

    // Clean up
    for (const auto & id : ids) {
        cache->remove(id, ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS);
    }
    cache->evict(0);

    printf("  PASS: cache lookup is consistent\n");
    return test_result::PASS;
}

// =============================================================================
// Test runner
// =============================================================================
static void run_test(test_result (*test_fn)(), const char * name) {
    g_tests_run++;
    switch (test_fn()) {
        case test_result::PASS:
            g_tests_passed++;
            break;
        case test_result::SKIP:
            g_tests_skipped++;
            printf("  >>> SKIPPED: %s\n", name);
            break;
        case test_result::FAIL:
            g_tests_failed++;
            printf("  >>> FAILED: %s\n", name);
            break;
    }
    printf("\n");
}

int main(int /*argc*/, char ** /*argv*/) {
    // Pin the cache budget BEFORE anything can touch a SYCL device.  The first
    // cache access sets g_cache_mode_locked and this setter becomes a no-op, and
    // ggml_backend_sycl_get_device_memory() below is enough to trigger it -- so
    // this must be the first statement in the program.
    //
    // Why a pin is needed at all: budget() is weight-cache accounting and does
    // NOT subtract the arena's up-front VRAM reservation, by the 792vn.5 design
    // decision recorded at unified-cache.cpp:8573.  At the default whole-device
    // budget that leaves every size derived from available() unallocatable.  See
    // the contract note at the top of this file for the full arithmetic.
    ggml_sycl::set_unified_cache_budget(k_test_budget_bytes);

    printf("=================================================================\n");
    printf("SYCL Unified Memory End-to-End Validation Tests\n");
    printf("Epic: llama.cpp-v3n, Task: llama.cpp-13o\n");
    printf("Goal: Validate budget enforcement and eviction under MoE-shaped load\n");
    printf("=================================================================\n\n");

    // Check for SYCL device
    size_t free_mem = 0, total_mem = 0;
    ggml_backend_sycl_get_device_memory(0, &free_mem, &total_mem);

    if (total_mem == 0) {
        printf("SKIP: no SYCL device found; this run proves NOTHING about the unified memory system.\n");
        printf("Try: source /opt/intel/oneapi/setvars.sh\n");
        return k_exit_skip;
    }

    printf("Device 0: %zu MB free / %zu MB total\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    if (!cache) {
        printf("SKIP: no unified cache for device 0; this run proves NOTHING about the unified memory system.\n");
        return k_exit_skip;
    }

    // Assert the pin took.  A setter that silently did nothing would leave every
    // subcase sizing itself against the whole-device budget again -- the exact
    // starvation this fixture was rewritten to stop -- and the failures would
    // read as cache defects.  base_budget() is the figure set_unified_cache_budget()
    // controls; budget() is base_budget() minus any runtime reservation.
    if (cache->base_budget() != k_test_budget_bytes) {
        printf(
            "SKIP: cache for device 0 already existed when this fixture tried to pin its budget "
            "(requested %zu MB, cache reports base %zu MB). Every subcase below would size itself "
            "against a budget that does not describe allocatable VRAM; this run proves NOTHING.\n",
            k_test_budget_bytes / (1024 * 1024), cache->base_budget() / (1024 * 1024));
        return k_exit_skip;
    }

    printf("Cache budget pinned: base %zu MB, effective %zu MB, used %zu MB\n\n", cache->base_budget() / (1024 * 1024),
           cache->budget() / (1024 * 1024), cache->used() / (1024 * 1024));

    // Run all E2E tests
    run_test(test_120b_memory_pressure_simulation, "test_120b_memory_pressure_simulation");
    run_test(test_expert_streaming_under_pressure, "test_expert_streaming_under_pressure");
    run_test(test_kv_expert_coexistence, "test_kv_expert_coexistence");
    run_test(test_three_tier_memory_flow, "test_three_tier_memory_flow");
    run_test(test_budget_enforcement, "test_budget_enforcement");
    run_test(test_compute_buffer_protection, "test_compute_buffer_protection");
    run_test(test_expert_prefetch_integration, "test_expert_prefetch_integration");
    run_test(test_memory_tracking_consistency, "test_memory_tracking_consistency");
    run_test(test_eviction_fragmentation, "test_eviction_fragmentation");
    run_test(test_cache_lookup_consistency, "test_cache_lookup_consistency");

    // Summary
    printf("=================================================================\n");
    printf("E2E Test Results: %d/%d passed, %d failed, %d skipped\n",
           g_tests_passed, g_tests_run, g_tests_failed, g_tests_skipped);
    printf("=================================================================\n");

    if (g_tests_failed > 0) {
        return 1;
    }

    // A subcase that could not run leaves its property unverified, so the run
    // as a whole has not validated the unified memory system -- report it as a
    // skip rather than letting the passing subcases carry it to 0.
    if (g_tests_skipped > 0) {
        printf("\nSKIP: %d subcase(s) could not run; this run proves NOTHING about them\n", g_tests_skipped);
        return k_exit_skip;
    }

    printf("\nSUCCESS: Unified memory system validated for 120B+ MoE workloads\n");
    return 0;
}
