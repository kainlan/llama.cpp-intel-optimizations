// Focused regression test for unified-cache MoE expert handle resolution.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-sycl-moe-handle-resolution

#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/mem-handle.hpp"
#include "ggml-sycl/unified-cache.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

#    define TEST_ASSERT(cond, msg)                    \
        do {                                          \
            if (!(cond)) {                            \
                fprintf(stderr, "  FAIL: %s\n", msg); \
                return false;                         \
            }                                         \
        } while (0)

static ggml_sycl::expert_resolve_request make_request(const ggml_sycl_cache_id & key,
                                                      ggml_layout_mode           layout,
                                                      int                        current_device = 0) {
    ggml_sycl::expert_resolve_request req{};
    req.key              = key;
    req.requested_layout = layout;
    req.layer_id         = 7;
    req.expert_id        = 3;
    req.current_device   = current_device;
    req.preferred_device = current_device;
    req.device_policy    = ggml_sycl::expert_resolve_device_policy::PREFER_CURRENT;
    req.allow_host       = true;
    req.allow_mmap_host  = true;
    return req;
}

static bool test_normal_cache_expert_resolution(sycl::queue & q) {
    printf("\n=== Test: normal cache expert resolution ===\n");

    ggml_sycl::unified_cache cache(q, 16 * 1024);
    std::vector<uint8_t>     data(128, 0x31);
    ggml_sycl_cache_id       key = ggml_sycl::test_make_cache_id(data.data());
    key.aux_id                   = 0x70003;

    void * ptr = cache.allocate_slot(key, data.size(), GGML_LAYOUT_AOS, ggml_sycl::cache_entry_type::MOE_EXPERT, 7, 3);
    TEST_ASSERT(ptr != nullptr, "allocate_slot failed");
    q.memcpy(ptr, data.data(), data.size()).wait();
    cache.register_ready(key, ptr, GGML_LAYOUT_AOS, data.size(), ggml_sycl::cache_entry_type::MOE_EXPERT, 7, 3,
                         data.data());

    auto res = cache.resolve_expert(make_request(key, GGML_LAYOUT_AOS));
    TEST_ASSERT(res.reason == ggml_sycl::expert_resolve_reason::FOUND, "normal expert should resolve");
    TEST_ASSERT(res.ptr == ptr, "normal expert pointer mismatch");
    TEST_ASSERT(res.tier == ggml_sycl::expert_resolve_tier::DEVICE_VRAM, "normal expert tier mismatch");
    TEST_ASSERT(res.location == ggml_sycl::cache_location::DEVICE, "normal expert location mismatch");
    TEST_ASSERT(res.owning_device == 0, "normal expert owner should be device 0 under level_zero:0");
    TEST_ASSERT(res.actual_layout == GGML_LAYOUT_AOS, "normal expert layout mismatch");
    TEST_ASSERT(!res.cpu_accessible, "device expert should not report CPU accessibility");
    TEST_ASSERT(res.lifetime != nullptr, "normal expert should return a lifetime handle");
    TEST_ASSERT(res.lifetime->resolve().ptr == ptr, "normal expert lifetime handle did not resolve");

    auto wrong_device_req          = make_request(key, GGML_LAYOUT_AOS, 1);
    wrong_device_req.device_policy = ggml_sycl::expert_resolve_device_policy::CURRENT_ONLY;
    auto wrong_device              = cache.resolve_expert(wrong_device_req);
    TEST_ASSERT(wrong_device.reason == ggml_sycl::expert_resolve_reason::DEVICE_MISMATCH,
                "current-device-only policy should reject another device");

    return true;
}

static bool test_direct_staged_device_resolution(sycl::queue & q) {
    printf("\n=== Test: direct staged device expert resolution ===\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    TEST_ASSERT(cache != nullptr, "global unified cache unavailable");
    std::vector<uint8_t> data(256, 0x42);
    ggml_sycl_cache_id   key = ggml_sycl::test_make_cache_id(data.data());
    key.aux_id               = 0x70004;

    auto stage =
        cache->direct_stage_expert(key, data.data(), data.size(), data.size(), GGML_LAYOUT_SOA, nullptr, nullptr, &q);
    TEST_ASSERT(stage.ok && stage.ptr != nullptr, "direct_stage_expert failed");

    auto res = cache->resolve_expert(make_request(key, GGML_LAYOUT_SOA));
    TEST_ASSERT(res.reason == ggml_sycl::expert_resolve_reason::FOUND, "direct staged expert should resolve");
    TEST_ASSERT(res.ptr == stage.ptr, "direct staged expert pointer mismatch");
    TEST_ASSERT(res.tier == ggml_sycl::expert_resolve_tier::DEVICE_VRAM, "direct staged expert tier mismatch");
    TEST_ASSERT(res.location == ggml_sycl::cache_location::DEVICE, "direct staged expert location mismatch");
    TEST_ASSERT(res.owning_device == 0, "direct staged expert owner should be device 0 under level_zero:0");
    TEST_ASSERT(res.actual_layout == GGML_LAYOUT_SOA, "direct staged expert layout mismatch");
    TEST_ASSERT(res.has_ready_event, "direct staged expert should expose ready event metadata");
    TEST_ASSERT(res.lifetime != nullptr, "direct staged expert should return a lifetime handle");
    stage.event.wait();

    return true;
}

static bool test_direct_host_and_miss_resolution(sycl::queue & q) {
    printf("\n=== Test: direct host and miss expert resolution ===\n");

    ggml_sycl::unified_cache cache(q, 16 * 1024);
    uint8_t *                pinned = static_cast<uint8_t *>(sycl::malloc_host(64, q));
    TEST_ASSERT(pinned != nullptr, "sycl::malloc_host failed");
    std::fill(pinned, pinned + 64, 0x53);
    ggml_sycl_cache_id host_key = ggml_sycl::test_make_cache_id(pinned);
    host_key.aux_id             = 0x70005;

    cache.register_host_expert(host_key, pinned, 64, GGML_LAYOUT_AOS);

    auto host_res = cache.resolve_expert(make_request(host_key, GGML_LAYOUT_AOS));
    TEST_ASSERT(host_res.reason == ggml_sycl::expert_resolve_reason::FOUND, "host expert should resolve");
    TEST_ASSERT(host_res.ptr == pinned, "host expert pointer mismatch");
    TEST_ASSERT(host_res.tier == ggml_sycl::expert_resolve_tier::HOST_PINNED, "host expert tier mismatch");
    TEST_ASSERT(host_res.location == ggml_sycl::cache_location::HOST_PINNED, "host expert location mismatch");
    TEST_ASSERT(host_res.owning_device == ggml_sycl::mem_handle::HOST_DEVICE, "host expert owner mismatch");
    TEST_ASSERT(host_res.cpu_accessible, "host expert should be CPU accessible");
    TEST_ASSERT(host_res.lifetime != nullptr, "host expert should return a lifetime handle");

    auto host_blocked_req       = make_request(host_key, GGML_LAYOUT_AOS);
    host_blocked_req.allow_host = false;
    auto host_blocked           = cache.resolve_expert(host_blocked_req);
    TEST_ASSERT(host_blocked.reason == ggml_sycl::expert_resolve_reason::HOST_DISALLOWED,
                "host-pinned expert should honor allow_host=false for device-planned routes");

    auto layout_miss = cache.resolve_expert(make_request(host_key, GGML_LAYOUT_SOA));
    TEST_ASSERT(layout_miss.reason == ggml_sycl::expert_resolve_reason::LAYOUT_MISMATCH,
                "wrong layout should report layout mismatch");

    std::vector<uint8_t> plain_host(64, 0x54);
    ggml_sycl_cache_id   mmap_key = ggml_sycl::test_make_cache_id(plain_host.data());
    mmap_key.aux_id               = 0x70007;
    cache.register_host_expert(mmap_key, plain_host.data(), plain_host.size(), GGML_LAYOUT_AOS);

    auto mmap_res = cache.resolve_expert(make_request(mmap_key, GGML_LAYOUT_AOS));
    TEST_ASSERT(mmap_res.reason == ggml_sycl::expert_resolve_reason::FOUND, "plain host expert should resolve");
    TEST_ASSERT(mmap_res.tier == ggml_sycl::expert_resolve_tier::HOST_MMAP, "plain host expert tier mismatch");
    TEST_ASSERT(mmap_res.location == ggml_sycl::cache_location::HOST_MMAP, "plain host expert location mismatch");
    TEST_ASSERT(mmap_res.cpu_accessible, "plain host expert should be CPU accessible");

    auto mmap_blocked_req            = make_request(mmap_key, GGML_LAYOUT_AOS);
    mmap_blocked_req.allow_mmap_host = false;
    auto mmap_blocked                = cache.resolve_expert(mmap_blocked_req);
    TEST_ASSERT(mmap_blocked.reason == ggml_sycl::expert_resolve_reason::MMAP_HOST_DISALLOWED,
                "plain host expert should honor allow_mmap_host=false");

    ggml_sycl_cache_id missing_key = ggml_sycl::test_make_cache_id(reinterpret_cast<void *>(0x12345));
    missing_key.aux_id             = 0x70006;
    auto missing                   = cache.resolve_expert(make_request(missing_key, GGML_LAYOUT_AOS));
    TEST_ASSERT(missing.reason == ggml_sycl::expert_resolve_reason::NOT_FOUND, "missing expert reason mismatch");
    TEST_ASSERT(missing.tier == ggml_sycl::expert_resolve_tier::UNAVAILABLE, "missing expert tier mismatch");
    TEST_ASSERT(missing.ptr == nullptr, "missing expert pointer should be null");

    sycl::free(pinned, q);
    return true;
}

static bool test_expert_staging_host_compute_zone_ownership(sycl::queue & q) {
    printf("\n=== Test: EXPERT_STAGING HOST_COMPUTE zone ownership ===\n");

    ggml_sycl::unified_cache * cache = ggml_sycl::get_unified_cache_for_device(0);
    TEST_ASSERT(cache != nullptr, "global unified cache unavailable");
    if (!cache->host_zones_configured()) {
        constexpr size_t mib = 1024u * 1024u;
        cache->configure_host_zones(8u * mib, 8u * mib, 8u * mib, 8u * mib);
    }
    TEST_ASSERT(cache->host_zones_configured(), "host zones should be configured for ownership test");

    const size_t scratch_before = cache->host_zone_used(ggml_sycl::host_zone_id::SCRATCH);

    ggml_sycl::alloc_request req{};
    req.queue                               = &q;
    req.device                              = 0;
    req.size                                = 256u * 1024u;
    req.intent.role                         = ggml_sycl::alloc_role::EXPERT_STAGING;
    req.intent.category                     = ggml_sycl::runtime_category::HOST_COMPUTE;
    req.intent.cohort_id                    = "test_moe_expert_staging";
    req.intent.constraints.must_host_pinned = true;
    req.intent.constraints.use_pinned_pool  = true;

    ggml_sycl::alloc_handle h{};
    TEST_ASSERT(ggml_sycl::unified_alloc(req, &h), "EXPERT_STAGING HOST_COMPUTE unified_alloc failed");
    TEST_ASSERT(h.ptr != nullptr, "EXPERT_STAGING HOST_COMPUTE returned null ptr");
    TEST_ASSERT(h.tier == ggml_sycl::alloc_tier::HOST_PINNED, "EXPERT_STAGING should resolve to host-pinned tier");
    TEST_ASSERT(h.zone_managed, "EXPERT_STAGING should be zone managed");
    TEST_ASSERT(h.host_zone == ggml_sycl::host_zone_id::SCRATCH, "EXPERT_STAGING should route to SCRATCH host zone");

    auto mh = h.as_mem_handle();
    auto r  = mh.resolve(0);
    TEST_ASSERT(r.ptr == h.ptr, "mem_handle resolved pointer mismatch");
    TEST_ASSERT(!r.on_device, "EXPERT_STAGING mem_handle should resolve as host accessible");
    const sycl::usm::alloc ptr_type = sycl::get_pointer_type(h.ptr, q.get_context());
    TEST_ASSERT(ptr_type == sycl::usm::alloc::host || ptr_type == sycl::usm::alloc::shared,
                "EXPERT_STAGING pointer should be SYCL host/shared USM");

    const size_t scratch_after_alloc = cache->host_zone_used(ggml_sycl::host_zone_id::SCRATCH);
    TEST_ASSERT(scratch_after_alloc > scratch_before, "SCRATCH zone usage should increase after allocation");
    TEST_ASSERT(ggml_sycl::unified_free(h), "EXPERT_STAGING unified_free failed");
    const size_t scratch_after_free = cache->host_zone_used(ggml_sycl::host_zone_id::SCRATCH);
    TEST_ASSERT(scratch_after_free == scratch_before, "EXPERT_STAGING SCRATCH allocation should be reclaimed by free");

    return true;
}

static bool test_planner_role_specific_expert_placement() {
    printf("\n=== Test: planner role-specific expert placement ===\n");

    constexpr int                                     n_experts = 32;
    const std::vector<std::pair<std::string, size_t>> inventory = {
        { "blk.0.ffn_gate_exps.weight", 512u * n_experts  },
        { "blk.0.ffn_up_exps.weight",   2048u * n_experts },
        { "blk.0.ffn_down_exps.weight", 1024u * n_experts },
        { "blk.8.ffn_gate_exps.weight", 512u * n_experts  },
        { "blk.8.ffn_up_exps.weight",   2048u * n_experts },
        { "blk.8.ffn_down_exps.weight", 1024u * n_experts },
    };
    constexpr size_t                            mib     = 1024u * 1024u;
    const std::vector<ggml_sycl::device_budget> devices = {
        { 0, 1024u * mib + 32u * 1024u, 1024u * mib + 32u * 1024u },
        { 1, 64u * 1024u,               64u * 1024u               },
    };

    ggml_sycl::placement_kv_info kv_info{};
    auto plan = ggml_sycl::compute_multi_device_plan(devices, inventory, 9, ggml_sycl::multi_gpu_mode::HYBRID, kv_info,
                                                     nullptr, n_experts);

    auto summary = plan.summarize_expert_placements(2, n_experts);
    TEST_ASSERT(summary.expected == 2u * n_experts * 3u, "expected count should cover all layer/expert/role tensors");
    TEST_ASSERT(summary.planned == summary.expected, "all expert role placements should be planned");
    TEST_ASSERT(summary.missing == 0, "no expert role placement should be missing");
    TEST_ASSERT(summary.unclassified == 0, "no expert role placement should be unclassified");
    TEST_ASSERT(summary.duplicates == 0, "no expert role placement should be duplicated");
    TEST_ASSERT(summary.role_counts[static_cast<size_t>(ggml_sycl::expert_tensor_role::GATE)] == 2u * n_experts,
                "gate role count mismatch");
    TEST_ASSERT(summary.role_counts[static_cast<size_t>(ggml_sycl::expert_tensor_role::UP)] == 2u * n_experts,
                "up role count mismatch");
    TEST_ASSERT(summary.role_counts[static_cast<size_t>(ggml_sycl::expert_tensor_role::DOWN)] == 2u * n_experts,
                "down role count mismatch");

    size_t target_total = 0;
    for (const auto & [target, count] : summary.target_counts) {
        (void) target;
        target_total += count;
    }
    TEST_ASSERT(target_total == summary.expected, "device/host target counts should sum to expert tensor count");
    TEST_ASSERT(summary.target_counts[-1] > 0, "small synthetic budget should spill some expert tensors to host");
    TEST_ASSERT(summary.target_counts[0] + summary.target_counts[1] > 0,
                "synthetic budget should place some expert tensors on device");
    TEST_ASSERT(plan.moe_cpu_expert_staging_bytes > 0, "planner should size MoE CPU expert staging");
    TEST_ASSERT(plan.host_zone_scratch_bytes >= plan.moe_cpu_expert_staging_bytes,
                "SCRATCH host zone should include MoE CPU expert staging");

    const auto gate = plan.lookup_expert_placement(8, 30, ggml_sycl::expert_tensor_role::GATE);
    const auto up   = plan.lookup_expert_placement(8, 30, ggml_sycl::expert_tensor_role::UP);
    const auto down = plan.lookup_expert_placement(8, 30, ggml_sycl::expert_tensor_role::DOWN);
    TEST_ASSERT(gate.found(), "blk.8 expert 30 gate placement should resolve");
    TEST_ASSERT(up.found(), "blk.8 expert 30 up placement should resolve");
    TEST_ASSERT(down.found(), "blk.8 expert 30 down placement should resolve");
    TEST_ASSERT(gate.tensor_name.find("ffn_gate_exps") != std::string::npos, "gate lookup returned wrong tensor");
    TEST_ASSERT(up.tensor_name.find("ffn_up_exps") != std::string::npos, "up lookup returned wrong tensor");
    TEST_ASSERT(down.tensor_name.find("ffn_down_exps") != std::string::npos, "down lookup returned wrong tensor");

    const auto missing = plan.lookup_expert_placement(8, 99, ggml_sycl::expert_tensor_role::GATE);
    TEST_ASSERT(!missing.found(), "missing expert placement should be explicit");
    TEST_ASSERT(!plan.expert_on_device("blk.8.ffn_gate_exps.weight", 99),
                "legacy bool helper must not treat missing expert as device-resident");

    ggml_sycl::placement_plan manual;
    manual.entries.push_back({ "blk.2.ffn_gate_exps.weight", 1024, 1024, 0,
                               ggml_sycl::placement_priority::MOE_GATE_PROJ, 2, 5, ggml_sycl::expert_tensor_role::GATE,
                               true, 1 });
    manual.entries.push_back({ "blk.2.ffn_up_exps.weight", 1024, 1024, 0, ggml_sycl::placement_priority::MOE_UP, 2, 5,
                               ggml_sycl::expert_tensor_role::UP, true, 0 });
    manual.entries.push_back({ "blk.2.ffn_down_exps.weight", 1024, 1024, 0, ggml_sycl::placement_priority::MOE_DOWN, 2,
                               5, ggml_sycl::expert_tensor_role::DOWN, false, -1 });
    manual.expert_device[2][5] = -1;  // Deliberately stale compatibility summary.
    manual.build_index();

    const auto manual_gate = manual.lookup_expert_placement("blk.2.ffn_gate_exps.weight", 5);
    const auto manual_up   = manual.lookup_expert_placement("blk.2.ffn_up_exps.weight", 5);
    const auto manual_down = manual.lookup_expert_placement("blk.2.ffn_down_exps.weight", 5);
    TEST_ASSERT(manual_gate.found() && manual_gate.on_device && manual_gate.target_device == 1,
                "role-specific gate placement must preserve its planned secondary device");
    TEST_ASSERT(manual_up.found() && manual_up.on_device && manual_up.target_device == 0,
                "role-specific up placement must preserve its planned primary device");
    TEST_ASSERT(manual_down.found() && !manual_down.on_device,
                "role-specific down placement must preserve host placement");
    TEST_ASSERT(manual.expert_on_device("blk.2.ffn_gate_exps.weight", 5, 1),
                "role-specific helper must ignore stale collapsed expert_device summary");
    TEST_ASSERT(!manual.expert_on_device("blk.2.ffn_gate_exps.weight", 5, 0),
                "role-specific helper must not route gate to the wrong device");

    return true;
}

static bool test_moe_route_preserves_ready_event_for_chaining() {
    printf("\n=== Test: MoE route ready-event chaining contract ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_route_preserves_ready_event_for_chaining(),
                "planned MoE route resolution should preserve ready_event instead of waiting");

    return true;
}

static bool test_moe_ptr_table_retains_route_lease_until_event() {
    printf("\n=== Test: MoE pointer-table route lease lifetime ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_ptr_table_retains_route_lease_until_event(),
                "update_moe_ptr_table should retain route leases and chain ready_event into table memcpy");

    return true;
}

static bool test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event() {
    printf("\n=== Test: MoE shared ID cache rebuilds pointer-table lease lifetime ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event(),
                "shared ID reuse should rebuild current-role pointer tables with leases and ready_event dependencies");

    return true;
}

static bool test_moe_ptr_table_cached_reuse_is_tensor_specific() {
    printf("\n=== Test: MoE shared ID cache rebuilds tensor-specific pointer tables ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_ptr_table_cached_reuse_is_tensor_specific(),
                "UP/DOWN pointer tables must be rebuilt from their own role handles, not GATE pointers");

    return true;
}

static bool test_moe_ptr_table_does_not_persist_pointer_cache() {
    printf("\n=== Test: MoE pointer-table cache stores IDs only ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_ptr_table_does_not_persist_pointer_cache(),
                "MoE block cache should retain shared IDs only, not a separate pointer-handle cache");

    return true;
}

static bool test_moe_ptr_table_lease_covers_populated_slots() {
    printf("\n=== Test: MoE pointer-table populated slots are covered by leases ===\n");

    TEST_ASSERT(ggml_sycl::test_moe_ptr_table_lease_covers_populated_slots(),
                "every populated MoE pointer-table slot should resolve from a retained mem_handle lease");

    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    ok &= test_normal_cache_expert_resolution(q);
    ok &= test_direct_staged_device_resolution(q);
    ok &= test_direct_host_and_miss_resolution(q);
    ok &= test_expert_staging_host_compute_zone_ownership(q);
    ok &= test_planner_role_specific_expert_placement();
    ok &= test_moe_route_preserves_ready_event_for_chaining();
    ok &= test_moe_ptr_table_retains_route_lease_until_event();
    ok &= test_moe_ptr_table_cached_reuse_retains_lease_and_ready_event();
    ok &= test_moe_ptr_table_cached_reuse_is_tensor_specific();
    ok &= test_moe_ptr_table_does_not_persist_pointer_cache();
    ok &= test_moe_ptr_table_lease_covers_populated_slots();

    printf("\nSYCL MoE handle resolution tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
