#include "ggml-sycl/ggml-sycl-test.hpp"

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

static int test_grouped_metadata_still_blocks_direct_final() {
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
    in.down_sum_final_index               = 11;
    in.glu_handle_device                  = true;
    in.final_handle_device                = true;
    in.down_weight_resident               = true;
    in.token_major_deterministic_metadata = false;

    auto out = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "grouped metadata must not activate direct-final");
    CHECK(std::strcmp(out.reason, "metadata") == 0, "grouped metadata reject reason must be metadata");
    return 0;
}

static int test_builds_token_major_entries_in_stable_order() {
    const int32_t ids[]     = { 3, 7, 4, 8 };
    const float   weights[] = { 0.25f, 0.75f, 0.40f, 0.60f };
    ggml_sycl::test_moe_token_major_metadata_entry entries[4]{};

    ggml_sycl::test_moe_token_major_metadata_input in{};
    in.ids          = ids;
    in.weights      = weights;
    in.n_ids        = 2;
    in.n_tokens     = 2;
    in.n_experts    = 16;
    in.ids_nb0      = sizeof(int32_t);
    in.ids_nb1      = 2 * sizeof(int32_t);
    in.weights_nb1  = sizeof(float);
    in.weights_nb2  = 2 * sizeof(float);
    in.out_entries  = entries;
    in.out_capacity = 4;

    auto out = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(out.accepted, "complete token-major metadata must accept");
    CHECK(std::strcmp(out.reason, "none") == 0, "accepted metadata reason must be none");
    CHECK(out.entries == 4, "metadata must contain n_tokens * n_ids entries");
    CHECK(out.signature != 0, "metadata signature must be non-zero");

    CHECK(entries[0].token == 0 && entries[0].rank == 0 && entries[0].expert == 3 && entries[0].slot == 0,
          "entry 0 must be token 0 rank 0");
    CHECK(entries[1].token == 0 && entries[1].rank == 1 && entries[1].expert == 7 && entries[1].slot == 2,
          "entry 1 must be token 0 rank 1");
    CHECK(entries[2].token == 1 && entries[2].rank == 0 && entries[2].expert == 4 && entries[2].slot == 1,
          "entry 2 must be token 1 rank 0");
    CHECK(entries[3].token == 1 && entries[3].rank == 1 && entries[3].expert == 8 && entries[3].slot == 3,
          "entry 3 must be token 1 rank 1");
    CHECK(std::fabs(entries[1].weight - 0.75f) < 0.00001f, "route weight must be copied");
    return 0;
}

static int test_rejects_bad_metadata_inputs() {
    const int32_t ids[]     = { 1, -1 };
    const float   weights[] = { 1.0f, 0.0f };
    ggml_sycl::test_moe_token_major_metadata_entry entries[2]{};

    ggml_sycl::test_moe_token_major_metadata_input in{};
    in.ids          = ids;
    in.weights      = weights;
    in.n_ids        = 2;
    in.n_tokens     = 1;
    in.n_experts    = 8;
    in.ids_nb0      = sizeof(int32_t);
    in.ids_nb1      = 2 * sizeof(int32_t);
    in.weights_nb1  = sizeof(float);
    in.weights_nb2  = 2 * sizeof(float);
    in.out_entries  = entries;
    in.out_capacity = 2;

    auto bad_expert = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!bad_expert.accepted && std::strcmp(bad_expert.reason, "expert") == 0,
          "negative expert id must reject");

    in.ids     = nullptr;
    auto no_ids = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!no_ids.accepted && std::strcmp(no_ids.reason, "ids") == 0, "missing ids must reject");

    in.ids              = ids;
    in.weights          = nullptr;
    in.require_weights  = true;
    auto no_weights     = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!no_weights.accepted && std::strcmp(no_weights.reason, "weights") == 0, "missing weights must reject");

    const int32_t good_ids[] = { 1, 2 };
    in.ids                  = good_ids;
    in.weights              = weights;
    in.out_capacity         = 1;
    auto small              = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!small.accepted && std::strcmp(small.reason, "capacity") == 0, "small output buffer must reject");

    const float nan_weights[] = { 1.0f, NAN };
    in.out_capacity           = 2;
    in.weights                = nan_weights;
    auto nan_out              = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!nan_out.accepted && std::strcmp(nan_out.reason, "weights") == 0, "non-finite weight must reject");
    return 0;
}

static int test_rejects_overflow_shapes() {
    const int32_t ids[]     = { 0, 1, 2, 3 };
    const float   weights[] = { 1.0f, 0.0f, 0.0f, 0.0f };
    ggml_sycl::test_moe_token_major_metadata_entry entries[4]{};

    ggml_sycl::test_moe_token_major_metadata_input in{};
    in.ids          = ids;
    in.weights      = weights;
    in.n_ids        = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    in.n_tokens     = 1;
    in.n_experts    = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 2;
    in.ids_nb0      = sizeof(int32_t);
    in.ids_nb1      = sizeof(int32_t);
    in.weights_nb1  = sizeof(float);
    in.weights_nb2  = sizeof(float);
    in.out_entries  = entries;
    in.out_capacity = 1;

    auto huge_dims = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!huge_dims.accepted && std::strcmp(huge_dims.reason, "shape") == 0,
          "dimensions larger than int32 metadata fields must reject as shape");

    in.n_ids        = 2;
    in.n_tokens     = 2;
    in.n_experts    = 8;
    in.ids_nb0      = std::numeric_limits<int64_t>::max();
    in.ids_nb1      = 1;
    in.out_capacity = 4;
    auto id_offset_overflow = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!id_offset_overflow.accepted && std::strcmp(id_offset_overflow.reason, "shape") == 0,
          "overflowing id byte offset must reject as shape");

    in.ids_nb0     = sizeof(int32_t);
    in.ids_nb1     = 2 * sizeof(int32_t);
    in.weights_nb1 = std::numeric_limits<int64_t>::max();
    in.weights_nb2 = 1;
    auto weight_offset_overflow = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!weight_offset_overflow.accepted && std::strcmp(weight_offset_overflow.reason, "shape") == 0,
          "overflowing weight byte offset must reject as shape");

    in.n_ids       = 2;
    in.n_tokens    = std::numeric_limits<int32_t>::max();
    in.weights_nb1 = sizeof(float);
    in.weights_nb2 = 2 * sizeof(float);
    in.out_capacity = 1;
    auto slot_overflow = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(!slot_overflow.accepted && std::strcmp(slot_overflow.reason, "shape") == 0,
          "slot values larger than int32 must reject as shape before capacity");
    return 0;
}

int main() {
    if (test_grouped_metadata_still_blocks_direct_final() != 0) {
        return 1;
    }
    if (test_builds_token_major_entries_in_stable_order() != 0) {
        return 1;
    }
    if (test_rejects_bad_metadata_inputs() != 0) {
        return 1;
    }
    if (test_rejects_overflow_shapes() != 0) {
        return 1;
    }
    std::puts("PASS: MoE token-major metadata policy");
    return 0;
}
