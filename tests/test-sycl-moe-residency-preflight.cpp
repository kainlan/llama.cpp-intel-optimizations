#include "ggml-sycl-test.hpp"
#include "moe-layer-plan.hpp"
#include "unified-cache.hpp"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond, msg)                             \
    do {                                             \
        if (!(cond)) {                               \
            std::fprintf(stderr, "FAIL: %s\n", msg); \
            return 1;                                \
        }                                            \
    } while (0)

struct stable_handle_fixture {
    alignas(64) float storage[16] = {};
    ggml_sycl::unified_cache_entry entry{};
    ggml_sycl::mem_handle          handle;

    explicit stable_handle_fixture(ggml_layout_mode layout = GGML_LAYOUT_XMX_TILED) {
        entry.in_use_count.store(1);
        ggml_sycl_cache_id id = ggml_sycl::test_make_cache_id(storage);
        handle                = ggml_sycl::mem_handle::from_weight_lease(id, 0, storage, layout, true, &entry);
        handle.set_debug_owner("preflight-required");
    }
};

static int test_preflight_accept_allows_optimized_launch() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::moe_residency_preflight_input input;
    input.layer          = 3;
    input.device         = 0;
    input.required_bytes = 1024;
    input.budget         = { 4096, 4096 };

    ggml_sycl::moe_residency_preflight_result result = ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(result.accepted, "accepted budget must pass preflight");
    CHECK(!result.fallback_required, "accepted budget must not require fallback");
    CHECK(result.optimized_launch_allowed, "accepted budget must allow optimized launch");
    CHECK(result.reason == ggml_sycl::residency_reject_reason::NONE, "accepted reason must be none");
    CHECK(result.bytes_requested == 1024, "preflight records requested bytes");
    return 0;
}

static int test_preflight_budget_rejects_to_fallback() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::moe_residency_preflight_input input;
    input.layer          = 3;
    input.device         = 0;
    input.required_bytes = 8192;
    input.budget         = { 4096, 16384 };

    ggml_sycl::moe_residency_preflight_result result = ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(!result.accepted, "oversized preflight must reject");
    CHECK(result.fallback_required, "rejected preflight must require fallback");
    CHECK(!result.optimized_launch_allowed, "rejected preflight must block optimized launch");
    CHECK(result.reason == ggml_sycl::residency_reject_reason::BUDGET,
          "total bytes beyond budget must reject as budget");
    CHECK(result.largest_free_block == 16384, "preflight records largest free block");
    return 0;
}

static int test_preflight_fragmentation_rejects_to_fallback() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::moe_residency_preflight_input input;
    input.layer          = 3;
    input.device         = 0;
    input.required_bytes = 8192;
    input.budget         = { 16384, 4096 };

    ggml_sycl::moe_residency_preflight_result result = ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(!result.accepted, "fragmented preflight must reject");
    CHECK(result.fallback_required, "fragmented preflight must require fallback");
    CHECK(!result.optimized_launch_allowed, "fragmented preflight must block optimized launch");
    CHECK(result.reason == ggml_sycl::residency_reject_reason::FRAGMENTATION,
          "enough total bytes but too-small largest block must reject as fragmentation");
    CHECK(result.largest_free_block == 4096, "preflight records fragmented largest free block");
    return 0;
}

static int test_preflight_accept_retains_required_handle() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    stable_handle_fixture stable(GGML_LAYOUT_XMX_TILED);

    ggml_sycl::moe_residency_preflight_input input;
    input.layer           = 3;
    input.device          = 0;
    input.required_bytes  = 0;
    input.required_layout = GGML_LAYOUT_XMX_TILED;
    input.required_handle = stable.handle;
    input.budget          = { 4096, 4096 };

    ggml_sycl::moe_residency_preflight_result result = ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(result.accepted, "resident handle preflight must accept");
    CHECK(result.retained_handle_count == 1, "accepted preflight must retain required handle");
    return 0;
}

static int test_preflight_rejects_raw_direct_required_handle() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    alignas(64) float storage[16] = {};

    ggml_sycl::moe_residency_preflight_input input;
    input.layer           = 3;
    input.device          = 0;
    input.required_bytes  = 0;
    input.required_layout = GGML_LAYOUT_XMX_TILED;
    input.required_handle = ggml_sycl::mem_handle::from_direct(storage, GGML_LAYOUT_XMX_TILED, false);
    input.budget          = { 4096, 4096 };

    ggml_sycl::moe_residency_preflight_result result = ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(!result.accepted, "raw direct required handle must reject preflight");
    CHECK(result.reason == ggml_sycl::residency_reject_reason::MISSING_LAYOUT,
          "raw direct required handle must reject as missing layout");
    return 0;
}

static int test_preflight_force_budget_env_is_opt_in() {
    ggml_sycl::moe_residency_preflight_input input;
    input.layer          = 3;
    input.device         = 0;
    input.required_bytes = 1024;
    input.budget         = { 4096, 4096 };

    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::moe_residency_preflight_result default_result =
        ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    CHECK(default_result.accepted, "default preflight must accept without force env");

    ::setenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES", "512", 1);
    ggml_sycl::moe_residency_preflight_result forced_result =
        ggml_sycl::evaluate_moe_residency_preflight_for_test(input);
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");

    CHECK(!forced_result.accepted, "forced budget env must force preflight fallback");
    CHECK(forced_result.fallback_required, "forced budget rejection must require fallback");
    CHECK(forced_result.reason == ggml_sycl::residency_reject_reason::BUDGET,
          "forced budget preflight must report budget reason");
    CHECK(default_result.optimized_launch_allowed, "default behavior remains optimized-launch eligible");
    return 0;
}

int main() {
    if (int rc = test_preflight_accept_allows_optimized_launch()) {
        return rc;
    }
    if (int rc = test_preflight_budget_rejects_to_fallback()) {
        return rc;
    }
    if (int rc = test_preflight_fragmentation_rejects_to_fallback()) {
        return rc;
    }
    if (int rc = test_preflight_accept_retains_required_handle()) {
        return rc;
    }
    if (int rc = test_preflight_rejects_raw_direct_required_handle()) {
        return rc;
    }
    if (int rc = test_preflight_force_budget_env_is_opt_in()) {
        return rc;
    }
    std::puts("PASS: MoE residency preflight policy");
    return 0;
}
