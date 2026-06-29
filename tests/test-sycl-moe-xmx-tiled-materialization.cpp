#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdio>

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
    std::puts("single-layout XMX_TILED materialization invariant tests passed");
    return 0;
}
