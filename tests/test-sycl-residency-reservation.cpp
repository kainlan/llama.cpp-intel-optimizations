#include "ggml-sycl-test.hpp"
#include "residency-plan.hpp"
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
        handle.set_debug_owner("reservation-handle");
    }
};

static ggml_sycl::residency_request make_request(size_t                bytes,
                                                 ggml_sycl::mem_handle handle         = {},
                                                 ggml_layout_mode      layout         = GGML_LAYOUT_XMX_TILED,
                                                 bool                  require_handle = false) {
    ggml_sycl::residency_request req;
    req.debug_name = "unit-reservation";
    req.device     = 0;
    req.phase      = ggml_sycl::residency_phase::MOE_DECODE;
    ggml_sycl::residency_entry_request entry;
    entry.tensor_name    = "gate";
    entry.layout         = layout;
    entry.bytes          = bytes;
    entry.role           = ggml_sycl::residency_role::MOE_GATE;
    entry.handle         = handle;
    entry.require_handle = require_handle;
    req.entries.push_back(entry);
    return req;
}

static int test_accept_retains_handles() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture stable(GGML_LAYOUT_XMX_TILED);

    ggml_sycl::residency_budget budget{ 4096, 4096 };
    ggml_sycl::residency_plan   plan = ggml_sycl::evaluate_residency_request_for_test(
        make_request(1024, stable.handle, GGML_LAYOUT_XMX_TILED, true), budget);

    CHECK(plan.accepted, "request should fit budget");
    CHECK(plan.reason == ggml_sycl::residency_reject_reason::NONE, "accepted request reason must be NONE");
    CHECK(plan.bytes_requested == 1024, "requested bytes must be recorded");
    CHECK(plan.bytes_reserved == 1024, "reserved bytes must match accepted request");
    CHECK(plan.entries.size() == 1, "accepted plan must return entries");
    CHECK(plan.entries[0].handle.valid(), "accepted plan must retain handle copy");
    CHECK(plan.entries[0].handle.stable_identity_equal(stable.handle),
          "retained handle must match request handle identity");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().accept_count == 1,
          "accepted reservation must increment diagnostics");
    return 0;
}

static int test_budget_rejects_before_entries() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::residency_diagnostics_reset_for_test();
    ggml_sycl::residency_budget budget{ 512, 4096 };
    ggml_sycl::residency_plan   plan = ggml_sycl::evaluate_residency_request_for_test(make_request(1024), budget);

    CHECK(!plan.accepted, "oversized request must reject");
    CHECK(plan.reason == ggml_sycl::residency_reject_reason::BUDGET, "reject reason must be budget");
    CHECK(plan.entries.empty(), "rejected plan must not materialize entries");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().reject_budget == 1,
          "budget rejection must increment diagnostics");
    return 0;
}

static int test_fragmentation_rejects_before_budget() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::residency_diagnostics_reset_for_test();
    ggml_sycl::residency_budget budget{ 4096, 512 };
    ggml_sycl::residency_plan   plan = ggml_sycl::evaluate_residency_request_for_test(make_request(1024), budget);

    CHECK(!plan.accepted, "fragmented request must reject");
    CHECK(plan.reason == ggml_sycl::residency_reject_reason::FRAGMENTATION, "reject reason must be fragmentation");
    CHECK(plan.entries.empty(), "fragmented rejected plan must not materialize entries");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().reject_fragmentation == 1,
          "fragmentation rejection must increment diagnostics");
    return 0;
}

static int test_missing_layout_rejects_requested_handle() {
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::residency_diagnostics_reset_for_test();
    stable_handle_fixture stable(GGML_LAYOUT_AOS);

    ggml_sycl::residency_budget budget{ 4096, 4096 };
    ggml_sycl::residency_plan   plan = ggml_sycl::evaluate_residency_request_for_test(
        make_request(0, stable.handle, GGML_LAYOUT_XMX_TILED, true), budget);

    CHECK(!plan.accepted, "required handle with wrong layout must reject");
    CHECK(plan.reason == ggml_sycl::residency_reject_reason::MISSING_LAYOUT,
          "wrong layout must reject as missing layout");
    CHECK(plan.entries.empty(), "missing-layout rejected plan must not retain entries");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().reject_missing_layout == 1,
          "missing-layout rejection must increment diagnostics");
    return 0;
}

static int test_forced_budget_env_is_opt_in_fallback() {
    ggml_sycl::residency_diagnostics_reset_for_test();
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");
    ggml_sycl::residency_budget budget{ 4096, 4096 };
    ggml_sycl::residency_plan default_plan = ggml_sycl::evaluate_residency_request_for_test(make_request(1024), budget);
    CHECK(default_plan.accepted, "without force env the request must accept");

    ggml_sycl::residency_diagnostics_reset_for_test();
    ::setenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES", "512", 1);
    ggml_sycl::residency_plan forced_plan = ggml_sycl::evaluate_residency_request_for_test(make_request(1024), budget);
    ::unsetenv("GGML_SYCL_RESIDENCY_FORCE_BUDGET_BYTES");

    CHECK(!forced_plan.accepted, "forced budget env must trigger fallback rejection");
    CHECK(forced_plan.reason == ggml_sycl::residency_reject_reason::BUDGET, "forced budget rejection must be BUDGET");
    CHECK(forced_plan.bytes_available == 512, "forced budget must cap available bytes");
    CHECK(ggml_sycl::residency_diagnostics_snapshot_for_test().reject_budget == 1,
          "forced fallback must record budget diagnostics");
    return 0;
}

int main() {
    if (int rc = test_accept_retains_handles()) {
        return rc;
    }
    if (int rc = test_budget_rejects_before_entries()) {
        return rc;
    }
    if (int rc = test_fragmentation_rejects_before_budget()) {
        return rc;
    }
    if (int rc = test_missing_layout_rejects_requested_handle()) {
        return rc;
    }
    if (int rc = test_forced_budget_env_is_opt_in_fallback()) {
        return rc;
    }
    std::puts("PASS: residency reservation policy");
    return 0;
}
