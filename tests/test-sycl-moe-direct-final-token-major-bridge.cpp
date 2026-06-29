#include "ggml-sycl/ggml-sycl-test.hpp"
#define GGML_SYCL_TEST_MOE_XMX_FUSED_HELPERS 1
#include "ggml-sycl/moe-xmx-fused.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static ggml_sycl::test_moe_down_sum_direct_final_input base_direct_final_input() {
    ggml_sycl::test_moe_down_sum_direct_final_input in{};
    in.env_enabled                        = true;
    in.n_tokens                           = 1;
    in.down_layout                        = GGML_LAYOUT_XMX_TILED;
    in.cached_q8_direct_ids               = true;
    in.has_ids_host                       = true;
    in.ids_host_count                     = 2;
    in.down_entries                       = 2;
    in.has_weighted_dst                   = true;
    in.has_moe_weights                    = true;
    in.has_down_sum_final                 = true;
    in.down_sum_final_index               = 44;
    in.glu_handle_device                  = true;
    in.final_handle_device                = true;
    in.down_weight_resident               = true;
    in.token_major_deterministic_metadata = true;
    in.token_major_entries                = 2;
    in.token_major_expected_entries       = 2;
    in.token_major_weights_valid          = true;
    return in;
}

static int test_token_major_metadata_accepts_direct_final_policy() {
    const int32_t ids[]     = { 5, 9 };
    const float   weights[] = { 0.60f, 0.40f };
    ggml_sycl::test_moe_token_major_metadata_entry entries[2]{};

    ggml_sycl::test_moe_token_major_metadata_input meta{};
    meta.ids          = ids;
    meta.weights      = weights;
    meta.n_ids        = 2;
    meta.n_tokens     = 1;
    meta.n_experts    = 16;
    meta.ids_nb0      = sizeof(int32_t);
    meta.ids_nb1      = 2 * sizeof(int32_t);
    meta.weights_nb1  = sizeof(float);
    meta.weights_nb2  = 2 * sizeof(float);
    meta.out_entries  = entries;
    meta.out_capacity = 2;

    auto meta_out = ggml_sycl::test_moe_build_token_major_metadata(meta);
    CHECK(meta_out.accepted, "token-major metadata must build");

    auto in = base_direct_final_input();
    in.token_major_entries          = meta_out.entries;
    in.token_major_expected_entries = in.down_entries;
    in.token_major_weights_valid    = meta_out.accepted;

    auto out = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(out.accepted, "complete token-major metadata must accept direct-final policy");
    CHECK(out.expected_saved_launches == 1, "accepted direct-final policy estimates one saved reduce launch");
    return 0;
}

static int test_incomplete_token_major_metadata_rejects_policy() {
    auto in                 = base_direct_final_input();
    in.token_major_entries  = 1;
    auto out                = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "incomplete token-major metadata must reject");
    CHECK(std::strcmp(out.reason, "metadata") == 0, "incomplete metadata reject reason must be metadata");
    return 0;
}

static int test_i8_and_dpas_require_complete_token_major_metadata() {
    auto in                               = base_direct_final_input();
    in.down_layout                       = GGML_LAYOUT_MXFP4_I8;
    in.env_enabled                       = false;
    in.dpas_direct_final_env_enabled     = true;
    in.dpas_direct_final_i8_enabled      = true;
    in.token_major_deterministic_metadata = false;
    auto i8                              = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!i8.accepted && std::strcmp(i8.reason, "metadata") == 0,
          "I8 direct-final must reject grouped metadata");

    in                                  = base_direct_final_input();
    in.down_layout                      = GGML_LAYOUT_MXFP4_DPAS;
    in.env_enabled                      = false;
    in.dpas_direct_final_env_enabled    = true;
    in.dpas_direct_final_dpas_enabled   = true;
    in.token_major_deterministic_metadata = false;
    auto dpas                           = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!dpas.accepted && std::strcmp(dpas.reason, "metadata") == 0,
          "DPAS direct-final must reject grouped metadata");

    in.token_major_deterministic_metadata = true;
    dpas                                  = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(dpas.accepted, "DPAS direct-final must accept complete deterministic metadata");
    return 0;
}

static int test_direct_final_requires_runtime_token_major_metadata() {
    ggml_sycl::test_moe_down_sum_direct_final_input in{};
    in.env_enabled                        = true;
    in.n_tokens                           = 1;
    in.down_layout                        = GGML_LAYOUT_XMX_TILED;
    in.cached_q8_direct_ids               = true;
    in.has_ids_host                       = true;
    in.ids_host_count                     = 8;
    in.down_entries                       = 8;
    in.has_weighted_dst                   = true;
    in.has_moe_weights                    = true;
    in.has_down_sum_final                 = true;
    in.down_sum_final_index               = 7;
    in.glu_handle_device                  = true;
    in.final_handle_device                = true;
    in.down_weight_resident               = true;
    in.token_major_deterministic_metadata = false;
    auto result = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!result.accepted, "grouped metadata must not enable direct-final");
    CHECK(std::strcmp(result.reason, "metadata") == 0, "grouped metadata reject reason must be stable");

    in.token_major_deterministic_metadata = true;
    in.token_major_entries                = 8;
    in.token_major_expected_entries       = 8;
    in.token_major_weights_valid          = true;
    result = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(result.accepted, "complete deterministic token-major metadata must enable direct-final policy");
    return 0;
}

static int test_token_major_metadata_view_requires_sorted_complete_weights() {
    const int32_t token_ids[]  = { 0, 0, 1, 1 };
    const int32_t expert_ids[] = { 3, 7, 4, 8 };
    const float   weights[]    = { 0.25f, 0.75f, 0.40f, 0.60f };
    ggml_sycl::test_moe_token_major_metadata_view view{};
    view.token_ids        = token_ids;
    view.expert_ids       = expert_ids;
    view.weights          = weights;
    view.entries          = 4;
    view.expected_entries = 4;
    CHECK(ggml_sycl::test_moe_token_major_metadata_is_complete(view),
          "sorted token-major metadata with valid weights must be complete");

    const int32_t unsorted_tokens[] = { 0, 1, 0, 1 };
    view.token_ids = unsorted_tokens;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "grouped/unsorted metadata must reject");

    const int32_t negative_tokens[] = { -1, 0, 1, 1 };
    view.token_ids = negative_tokens;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "negative token IDs must reject");

    view.token_ids = token_ids;
    view.entries   = 3;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "missing token-major entries must reject");

    const int32_t negative_experts[] = { 3, -7, 4, 8 };
    view.entries    = 4;
    view.expert_ids = negative_experts;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "negative expert IDs must reject");

    view.expert_ids = expert_ids;
    const float nan_weights[] = { 0.25f, NAN, 0.40f, 0.60f };
    view.weights = nan_weights;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "NaN weights must reject");

    const float inf_weights[] = { 0.25f, std::numeric_limits<float>::infinity(), 0.40f, 0.60f };
    view.weights = inf_weights;
    CHECK(!ggml_sycl::test_moe_token_major_metadata_is_complete(view), "+inf weights must reject");
    return 0;
}

static int test_prompt_rejects_before_metadata_acceptance() {
    auto in                         = base_direct_final_input();
    in.n_tokens                     = 2;
    in.down_entries                 = 4;
    in.ids_host_count               = 4;
    in.token_major_entries          = 4;
    in.token_major_expected_entries = 4;
    auto out                        = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "prompt direct-final policy must reject");
    CHECK(std::strcmp(out.reason, "tokens") == 0, "prompt reject reason must be tokens");
    return 0;
}

int main() {
    if (test_token_major_metadata_accepts_direct_final_policy() != 0) {
        return 1;
    }
    if (test_incomplete_token_major_metadata_rejects_policy() != 0) {
        return 1;
    }
    if (test_direct_final_requires_runtime_token_major_metadata() != 0) {
        return 1;
    }
    if (test_i8_and_dpas_require_complete_token_major_metadata() != 0) {
        return 1;
    }
    if (test_token_major_metadata_view_requires_sorted_complete_weights() != 0) {
        return 1;
    }
    if (test_prompt_rejects_before_metadata_acceptance() != 0) {
        return 1;
    }
    std::puts("PASS: MoE direct-final token-major bridge policy");
    return 0;
}
