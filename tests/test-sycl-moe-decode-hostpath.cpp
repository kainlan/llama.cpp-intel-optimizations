// Host-side decisions of the batch-1 (decode) MUL_MAT_ID path (llama.cpp-yx28).
// ggml-sycl/moe-decode-hostpath.hpp is SYCL-free, so these run without a device.
#include "ggml-sycl/moe-decode-hostpath.hpp"

#include <cstdio>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

static ggml_sycl::moe_decode_direct_request decode_request() {
    ggml_sycl::moe_decode_direct_request r;
    r.src1_tokens  = 1;
    r.ids_tokens   = 1;
    r.ids_selected = 10;
    return r;
}

static int test_direct_request() {
    using ggml_sycl::moe_decode_direct_request_admissible;
    ggml_sycl::moe_decode_direct_request r = decode_request();
    CHECK(moe_decode_direct_request_admissible(r), "a batch-1 decode op is admissible");

    r             = decode_request();
    r.src1_tokens = 2;
    CHECK(!moe_decode_direct_request_admissible(r), "a prompt batch never takes the decode direct route");
    r            = decode_request();
    r.ids_tokens = 4;
    CHECK(!moe_decode_direct_request_admissible(r), "more than one routed token is not decode");
    r              = decode_request();
    r.ids_selected = 0;
    CHECK(!moe_decode_direct_request_admissible(r), "no selected experts means nothing to dispatch");
    r                 = decode_request();
    r.graph_recording = true;
    CHECK(!moe_decode_direct_request_admissible(r), "graph recording keeps its own MoE handling");
    r                 = decode_request();
    r.layout_override = true;
    CHECK(!moe_decode_direct_request_admissible(r), "a diagnostic layout override keeps the retained route");
    r                        = decode_request();
    r.dedicated_decode_route = true;
    CHECK(!moe_decode_direct_request_admissible(r), "types with their own decode executor are not rerouted");
    return 0;
}

static int test_direct_stamp() {
    using ggml_sycl::moe_decode_direct_stamp_current;
    ggml_sycl::moe_decode_direct_stamp s{};
    CHECK(!moe_decode_direct_stamp_current(s, 1, 1, 10), "a never-evaluated tensor must be evaluated");

    ggml_sycl::moe_decode_direct_stamp_record(s, 3, 7, 10, /*layout=*/2, /*eligible=*/true);
    CHECK(moe_decode_direct_stamp_current(s, 3, 7, 10), "same generations and rows reuse the decision");
    CHECK(s.eligible && s.layout == 2, "the recorded decision and layout are kept");
    CHECK(!moe_decode_direct_stamp_current(s, 4, 7, 10), "a replan must re-evaluate residency");
    CHECK(!moe_decode_direct_stamp_current(s, 3, 8, 10), "an expert storage rewrite must re-evaluate");
    CHECK(!moe_decode_direct_stamp_current(s, 3, 7, 8), "a different selected-row count must re-evaluate");

    // An ineligible decision is cached too: a tensor with host experts is not
    // re-probed on every decode op of the same generation.
    ggml_sycl::moe_decode_direct_stamp_record(s, 3, 9, 10, /*layout=*/0, /*eligible=*/false);
    CHECK(moe_decode_direct_stamp_current(s, 3, 9, 10) && !s.eligible, "an ineligible decision is cached");
    return 0;
}

int main() {
    if (test_direct_request() != 0 || test_direct_stamp() != 0) {
        return 1;
    }
    std::printf("OK: moe decode host path decisions\n");
    return 0;
}
