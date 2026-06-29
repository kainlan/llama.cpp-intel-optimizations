#include "ggml-sycl/common.hpp"
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

static ggml_sycl::mxfp4_moe_single_gateup_layout_policy_input base_input() {
    ggml_sycl::mxfp4_moe_single_gateup_layout_policy_input in{};
    in.env_value        = "1";
    in.type             = GGML_TYPE_MXFP4;
    in.role             = ggml_sycl::expert_tensor_role::GATE;
    in.requested_layout = GGML_LAYOUT_XMX_TILED;
    in.device_resident  = true;
    in.xmx_int8_ok      = true;
    in.shape_aligned    = true;
    in.pp_rows          = 2048;
    in.pp_supported     = true;
    in.tg_supported     = true;
    return in;
}

static int test_default_off() {
    auto in      = base_input();
    in.env_value = nullptr;
    auto out     = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted, "unset env must reject");
    CHECK(std::strcmp(out.reason, "env") == 0, "unset env reason must be env");
    CHECK(out.requires_soa_alternate, "env-off must preserve current SOA safety behavior");

    in           = base_input();
    in.env_value = "0";
    out          = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted, "zero env must reject");
    CHECK(std::strcmp(out.reason, "env") == 0, "zero env reason must be env");
    return 0;
}

static int test_accepts_gate_and_up_when_pp_and_tg_are_supported() {
    auto gate = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(base_input());
    CHECK(gate.accepted, "gate must accept when all proof facts are true");
    CHECK(gate.layout == GGML_LAYOUT_XMX_TILED, "accepted layout must be XMX_TILED");
    CHECK(!gate.requires_soa_alternate, "accepted single-layout proof must not require SOA alternate");
    CHECK(std::strcmp(gate.route_label, "xmx-tiled-single-gateup") == 0, "route label must be stable");

    auto in = base_input();
    in.role = ggml_sycl::expert_tensor_role::UP;
    auto up = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(up.accepted, "up must accept under same proof facts");
    CHECK(!up.requires_soa_alternate, "up must not require SOA alternate when accepted");
    return 0;
}

static int test_rejects_unsafe_roles_shapes_and_missing_proofs() {
    auto in  = base_input();
    in.role  = ggml_sycl::expert_tensor_role::DOWN;
    auto out = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "role") == 0, "down role must reject");

    in      = base_input();
    in.type = GGML_TYPE_Q4_0;
    out     = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "type") == 0, "non-MXFP4 type must reject");

    in               = base_input();
    in.shape_aligned = false;
    out              = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "shape") == 0, "unaligned shape must reject");

    in              = base_input();
    in.pp_supported = false;
    out             = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "pp") == 0, "missing PP proof must reject");
    CHECK(out.requires_soa_alternate, "missing PP proof must preserve SOA safety fallback");

    in              = base_input();
    in.tg_supported = false;
    out             = ggml_sycl::mxfp4_moe_single_gateup_layout_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "tg") == 0, "missing TG proof must reject");
    return 0;
}

int main() {
    if (test_default_off() != 0) {
        return 1;
    }
    if (test_accepts_gate_and_up_when_pp_and_tg_are_supported() != 0) {
        return 1;
    }
    if (test_rejects_unsafe_roles_shapes_and_missing_proofs() != 0) {
        return 1;
    }
    std::puts("single-layout XMX_TILED gate/up policy tests passed");
    return 0;
}
