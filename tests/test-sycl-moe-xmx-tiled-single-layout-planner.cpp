#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static ggml_sycl::test_moe_single_xmx_planner_input base_input() {
    ggml_sycl::test_moe_single_xmx_planner_input in{};
    in.single_xmx_env               = "1";
    in.role                         = ggml_sycl::expert_tensor_role::GATE;
    in.type                         = GGML_TYPE_MXFP4;
    in.current_layout               = GGML_LAYOUT_XMX_TILED;
    in.device_resident              = true;
    in.device_xmx_int8_ok           = true;
    in.shape_aligned                = true;
    in.pp_rows                      = 2048;
    in.pp_xmx_supported             = true;
    in.tg_xmx_supported             = true;
    in.current_default_wants_pp_soa = true;
    return in;
}

static int test_default_keeps_pp_soa_rewrite() {
    auto in           = base_input();
    in.single_xmx_env = nullptr;
    auto out          = ggml_sycl::test_moe_single_xmx_planner_decision(in);
    CHECK(out.needs_pp_soa, "default path must keep PP SOA protection");
    CHECK(out.adds_soa_alternate, "default path may still add SOA alternate/protection");
    CHECK(!out.single_xmx_selected, "default path must not select single XMX mode");
    CHECK(std::strcmp(out.reason, "env") == 0, "default rejection reason must be env");
    return 0;
}

static int test_opt_in_suppresses_pp_soa_for_gate_and_up() {
    auto gate = ggml_sycl::test_moe_single_xmx_planner_decision(base_input());
    CHECK(gate.single_xmx_selected, "gate must select single XMX mode");
    CHECK(!gate.needs_pp_soa, "gate must not be rewritten to SOA in proof mode");
    CHECK(!gate.adds_soa_alternate, "gate must not get a full SOA alternate in proof mode");
    CHECK(std::strcmp(gate.reason, "none") == 0, "accepted reason must be none");

    auto in = base_input();
    in.role = ggml_sycl::expert_tensor_role::UP;
    auto up = ggml_sycl::test_moe_single_xmx_planner_decision(in);
    CHECK(up.single_xmx_selected, "up must select single XMX mode");
    CHECK(!up.needs_pp_soa, "up must not be rewritten to SOA in proof mode");
    CHECK(!up.adds_soa_alternate, "up must not get a full SOA alternate in proof mode");
    return 0;
}

static int test_down_and_missing_pp_proof_stay_safe() {
    auto in   = base_input();
    in.role   = ggml_sycl::expert_tensor_role::DOWN;
    auto down = ggml_sycl::test_moe_single_xmx_planner_decision(in);
    CHECK(!down.single_xmx_selected, "down must not use gate/up single-layout mode");
    CHECK(down.needs_pp_soa, "down follows existing safety decision in this helper input");

    in                  = base_input();
    in.pp_xmx_supported = false;
    auto no_pp          = ggml_sycl::test_moe_single_xmx_planner_decision(in);
    CHECK(!no_pp.single_xmx_selected, "missing PP proof must reject single-layout mode");
    CHECK(no_pp.needs_pp_soa, "missing PP proof must keep PP SOA protection");
    CHECK(std::strcmp(no_pp.reason, "pp") == 0, "missing PP proof reason must be pp");
    return 0;
}

static int test_pp_runtime_optins_require_forced_and_unsafe() {
    CHECK(!ggml_sycl::test_moe_single_xmx_pp_runtime_optins_from_env(nullptr, nullptr, nullptr),
          "default env must not allow prompt XMX proof");
    CHECK(!ggml_sycl::test_moe_single_xmx_pp_runtime_optins_from_env("1", nullptr, nullptr),
          "forced prompt XMX alone must not bypass submit watchdog");
    CHECK(!ggml_sycl::test_moe_single_xmx_pp_runtime_optins_from_env(nullptr, "1", nullptr),
          "unsafe PP alone must not force prompt XMX selection");
    CHECK(ggml_sycl::test_moe_single_xmx_pp_runtime_optins_from_env("1", "1", nullptr),
          "forced prompt XMX plus unsafe PP must allow diagnostic proof");
    CHECK(ggml_sycl::test_moe_single_xmx_pp_runtime_optins_from_env("1", nullptr, "1"),
          "legacy unsafe PP knob must still be honored with forced prompt XMX");
    return 0;
}

int main() {
    if (test_default_keeps_pp_soa_rewrite() != 0) {
        return 1;
    }
    if (test_opt_in_suppresses_pp_soa_for_gate_and_up() != 0) {
        return 1;
    }
    if (test_down_and_missing_pp_proof_stay_safe() != 0) {
        return 1;
    }
    if (test_pp_runtime_optins_require_forced_and_unsafe() != 0) {
        return 1;
    }
    std::puts("single-layout XMX_TILED planner tests passed");
    return 0;
}
