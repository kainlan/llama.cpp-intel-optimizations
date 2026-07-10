#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdio>
#include <string>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static int test_complete_xmx_handle_set() {
    ggml_sycl::test_moe_xmx_tiled_materialization_input in{};
    in.n_experts                 = 32;
    in.expected_layout           = GGML_LAYOUT_XMX_TILED;
    in.handles_present           = 32;
    in.handles_device_resident   = 32;
    in.handles_matching_layout   = 32;
    in.ready_events_present      = 32;
    in.single_xmx_mode           = true;
    in.materialization_succeeded = true;
    auto out                     = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(in);
    CHECK(out.complete, "complete handle set must be accepted");
    CHECK(out.release_soa_after_xmx, "single-layout success must release SOA source");
    CHECK(out.reason_code == 0, "success reason code must be zero");
    return 0;
}

static int test_incomplete_sets_reject() {
    auto base                      = ggml_sycl::test_moe_xmx_tiled_materialization_input{};
    base.n_experts                 = 32;
    base.expected_layout           = GGML_LAYOUT_XMX_TILED;
    base.handles_present           = 32;
    base.handles_device_resident   = 32;
    base.handles_matching_layout   = 32;
    base.ready_events_present      = 32;
    base.single_xmx_mode           = true;
    base.materialization_succeeded = true;

    auto missing            = base;
    missing.handles_present = 31;
    auto out                = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(missing);
    CHECK(!out.complete && out.reason_code == 1, "missing handle must reject");

    auto host                    = base;
    host.handles_device_resident = 31;
    out                          = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(host);
    CHECK(!out.complete && out.reason_code == 2, "host handle must reject");

    auto wrong_layout                    = base;
    wrong_layout.handles_matching_layout = 31;
    out                                  = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(wrong_layout);
    CHECK(!out.complete && out.reason_code == 3, "wrong layout must reject");

    auto no_event                 = base;
    no_event.ready_events_present = 31;
    out                           = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(no_event);
    CHECK(!out.complete && out.reason_code == 4, "missing ready event must reject");
    return 0;
}

static int test_no_soa_release_until_success() {
    ggml_sycl::test_moe_xmx_tiled_materialization_input in{};
    in.n_experts                 = 32;
    in.expected_layout           = GGML_LAYOUT_XMX_TILED;
    in.handles_present           = 32;
    in.handles_device_resident   = 32;
    in.handles_matching_layout   = 32;
    in.ready_events_present      = 32;
    in.single_xmx_mode           = true;
    in.materialization_succeeded = false;
    auto out                     = ggml_sycl::test_moe_xmx_tiled_materialization_invariants(in);
    CHECK(!out.complete, "failed materialization must not be complete");
    CHECK(!out.release_soa_after_xmx, "failed materialization must not release SOA source");
    return 0;
}

static int test_phase_xmx_auto_policy() {
    ggml_sycl::test_moe_phase_xmx_auto_policy_input in{};
    in.has_placement_plan = true;
    in.pp_soa_promoted    = true;
    in.xmx_supported      = true;
    in.xmx_int8_supported = true;

    auto out = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(out.materialization_enabled, "planner-promoted XMX decode must auto-enable phase materialization");
    CHECK(out.bulk_xmx_enabled, "planner-promoted XMX decode must auto-enable transactional bulk materialization");
    CHECK(std::string(out.reason) == "auto-capability", "automatic selection must report capability reason");

    in.has_placement_plan = false;
    out                   = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(!out.materialization_enabled && !out.bulk_xmx_enabled,
          "automatic selection must reject without a placement plan");

    in.has_placement_plan = true;
    in.pp_soa_promoted    = false;
    out                   = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(!out.materialization_enabled && !out.bulk_xmx_enabled,
          "automatic selection must reject a plan without PP-safe SOA promotion");

    in.pp_soa_promoted = true;
    in.xmx_supported   = false;
    out                = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(!out.materialization_enabled && !out.bulk_xmx_enabled,
          "automatic selection must reject a device without XMX");

    in.xmx_supported      = true;
    in.xmx_int8_supported = false;
    out                   = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(!out.materialization_enabled && !out.bulk_xmx_enabled,
          "automatic selection must reject XMX without INT8 support");

    in.xmx_int8_supported    = true;
    in.phase_materialize_env = "0";
    in.bulk_xmx_env          = nullptr;
    out                      = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(!out.materialization_enabled && out.bulk_xmx_enabled,
          "phase disable must override automatic execution without changing the unused bulk policy");
    CHECK(std::string(out.reason) == "phase-disabled", "phase disable must report a stable reason");

    in.phase_materialize_env = nullptr;
    in.bulk_xmx_env          = "0";
    out                      = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(out.materialization_enabled && !out.bulk_xmx_enabled,
          "bulk disable must preserve automatic phase selection and choose the per-expert fallback");
    CHECK(std::string(out.reason) == "bulk-disabled", "bulk disable must report a stable reason");

    in.has_placement_plan    = false;
    in.pp_soa_promoted       = false;
    in.xmx_supported         = false;
    in.xmx_int8_supported    = false;
    in.phase_materialize_env = "1";
    in.bulk_xmx_env          = "1";
    out                      = ggml_sycl::test_moe_phase_xmx_auto_policy(in);
    CHECK(out.materialization_enabled && out.bulk_xmx_enabled,
          "explicit diagnostic enables must remain available without automatic eligibility");
    CHECK(std::string(out.reason) == "diagnostic-override", "explicit opt-in must report diagnostic reason");
    return 0;
}

static int test_chunked_fallback_policy() {
    CHECK(
        ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("1", "blk.3.ffn_gate_exps.weight", GGML_LAYOUT_XMX_TILED,
                                                               /*bulk_materialization_ok=*/false),
        "opt-in gate XMX_TILED bulk failure should allow chunked fallback");
    CHECK(ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("1", "blk.3.ffn_up_exps.weight", GGML_LAYOUT_XMX_TILED,
                                                                 /*bulk_materialization_ok=*/false),
          "opt-in up XMX_TILED bulk failure should allow chunked fallback");
    CHECK(!ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("0", "blk.3.ffn_gate_exps.weight",
                                                                  GGML_LAYOUT_XMX_TILED,
                                                                  /*bulk_materialization_ok=*/false),
          "default-off mode must not allow chunked fallback");
    CHECK(!ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("1", "blk.3.ffn_down_exps.weight",
                                                                  GGML_LAYOUT_XMX_TILED,
                                                                  /*bulk_materialization_ok=*/false),
          "down tensors must not use gate/up single-XMX fallback");
    CHECK(!ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("1", "blk.3.ffn_gate_exps.weight", GGML_LAYOUT_SOA,
                                                                  /*bulk_materialization_ok=*/false),
          "non-XMX target must not use chunked fallback");
    CHECK(!ggml_sycl::test_moe_single_xmx_chunked_fallback_policy("1", "blk.3.ffn_gate_exps.weight",
                                                                  GGML_LAYOUT_XMX_TILED,
                                                                  /*bulk_materialization_ok=*/true),
          "successful bulk materialization must not report fallback");
    return 0;
}

int main() {
    if (test_complete_xmx_handle_set() != 0) {
        return 1;
    }
    if (test_incomplete_sets_reject() != 0) {
        return 1;
    }
    if (test_no_soa_release_until_success() != 0) {
        return 1;
    }
    if (test_chunked_fallback_policy() != 0) {
        return 1;
    }
    if (test_phase_xmx_auto_policy() != 0) {
        return 1;
    }
    std::puts("single-layout XMX_TILED materialization invariant tests passed");
    return 0;
}
