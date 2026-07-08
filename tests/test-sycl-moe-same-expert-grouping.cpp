#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static std::string join_path(const std::string & root, const char * rel) {
    if (root.empty() || root == ".") {
        return rel;
    }
    return root.back() == '/' ? root + rel : root + "/" + rel;
}

static std::vector<std::string> candidate_roots() {
    std::vector<std::string> roots;
    if (const char * env = std::getenv("LLAMA_CPP_REPO_ROOT")) {
        roots.emplace_back(env);
    }
    const std::string source_file = __FILE__;
    const std::string suffix      = "/tests/test-sycl-moe-same-expert-grouping.cpp";
    const size_t      pos         = source_file.rfind(suffix);
    if (pos != std::string::npos) {
        roots.emplace_back(source_file.substr(0, pos));
    }
    roots.emplace_back(".");
    roots.emplace_back("..");
    roots.emplace_back("../..");
    roots.emplace_back("../../..");
    return roots;
}

static std::string read_required_file(const char * rel) {
    for (const std::string & root : candidate_roots()) {
        std::ifstream in(join_path(root, rel), std::ios::binary);
        if (!in.good()) {
            continue;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
    std::fprintf(stderr, "FAIL: could not read required source file: %s\n", rel);
    std::exit(1);
}

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

static int build_token_major_entries(const int32_t *                                               ids,
                                     const float *                                                 weights,
                                     int64_t                                                       n_ids,
                                     int64_t                                                       n_tokens,
                                     int64_t                                                       n_experts,
                                     std::vector<ggml_sycl::test_moe_token_major_metadata_entry> & entries) {
    entries.assign(static_cast<size_t>(n_ids * n_tokens), {});
    ggml_sycl::test_moe_token_major_metadata_input in{};
    in.ids          = ids;
    in.weights      = weights;
    in.n_ids        = n_ids;
    in.n_tokens     = n_tokens;
    in.n_experts    = n_experts;
    in.ids_nb0      = sizeof(int32_t);
    in.ids_nb1      = n_ids * sizeof(int32_t);
    in.weights_nb1  = sizeof(float);
    in.weights_nb2  = n_ids * sizeof(float);
    in.out_entries  = entries.data();
    in.out_capacity = static_cast<int64_t>(entries.size());
    auto out        = ggml_sycl::test_moe_build_token_major_metadata(in);
    CHECK(out.accepted, "token-major metadata must build for grouping tests");
    CHECK(out.entries == static_cast<int64_t>(entries.size()), "token-major entry count mismatch");
    return 0;
}

static int test_groups_same_expert_entries_stably() {
    const int32_t                                               ids[]     = { 3, 5, 3, 3 };
    const float                                                 weights[] = { 0.25f, 0.5f, 0.125f, 0.125f };
    std::vector<ggml_sycl::test_moe_token_major_metadata_entry> entries;
    if (build_token_major_entries(ids, weights, 4, 1, 8, entries) != 0) {
        return 1;
    }

    ggml_sycl::test_moe_same_expert_group          groups[4]{};
    int32_t                                        grouped_indices[4]{};
    ggml_sycl::test_moe_same_expert_grouping_input in{};
    in.entries            = entries.data();
    in.entry_count        = static_cast<int64_t>(entries.size());
    in.expected_entries   = static_cast<int64_t>(entries.size());
    in.n_tokens           = 1;
    in.n_experts          = 8;
    in.max_lanes          = 8;
    in.out_groups         = groups;
    in.out_group_capacity = 4;
    in.out_entry_indices  = grouped_indices;
    in.out_entry_capacity = 4;
    auto out              = ggml_sycl::test_moe_build_same_expert_grouping(in);

    CHECK(out.accepted, "same-expert grouping must accept when at least one group fills multiple lanes");
    CHECK(out.has_lane_filled_group, "same-expert grouping must report lane-filled group");
    CHECK(out.groups == 2, "expected two groups keyed by first-seen expert order");
    CHECK(out.grouped_entries == 4, "all entries must be covered exactly once");
    CHECK(out.max_group_lanes == 3, "expert 3 must fill three DPAS lanes");

    CHECK(groups[0].token == 0 && groups[0].expert == 3 && groups[0].offset == 0 && groups[0].count == 3,
          "first group must be token 0 expert 3 with three stable entries");
    CHECK(grouped_indices[0] == 0 && grouped_indices[1] == 2 && grouped_indices[2] == 3,
          "expert 3 group must preserve rank order without mixing experts");
    CHECK(groups[1].token == 0 && groups[1].expert == 5 && groups[1].offset == 3 && groups[1].count == 1,
          "second group must be the singleton expert 5 group");
    CHECK(grouped_indices[3] == 1, "expert 5 singleton must reference rank 1 entry");

    for (int64_t g = 0; g < out.groups; ++g) {
        for (int32_t i = 0; i < groups[g].count; ++i) {
            const int32_t entry_index = grouped_indices[groups[g].offset + i];
            CHECK(entries[entry_index].expert == groups[g].expert, "a group must contain exactly one expert id");
            CHECK(entries[entry_index].token == groups[g].token, "a group must contain exactly one token id");
        }
    }
    return 0;
}

static int test_rejects_incomplete_or_unsorted_metadata() {
    ggml_sycl::test_moe_token_major_metadata_entry entries[2]{};
    entries[0] = { 0, 1, 2, 1, 0.5f };
    entries[1] = { 0, 0, 2, 0, 0.5f };
    ggml_sycl::test_moe_same_expert_group          groups[2]{};
    int32_t                                        grouped_indices[2]{};
    ggml_sycl::test_moe_same_expert_grouping_input in{};
    in.entries            = entries;
    in.entry_count        = 2;
    in.expected_entries   = 2;
    in.n_tokens           = 1;
    in.n_experts          = 4;
    in.max_lanes          = 8;
    in.out_groups         = groups;
    in.out_group_capacity = 2;
    in.out_entry_indices  = grouped_indices;
    in.out_entry_capacity = 2;
    auto unsorted         = ggml_sycl::test_moe_build_same_expert_grouping(in);
    CHECK(!unsorted.accepted, "unsorted token-major metadata must reject");
    CHECK(unsorted.reason && std::string(unsorted.reason) == "metadata", "unsorted reject reason must be metadata");

    entries[0]          = { 0, 0, 2, 0, 0.5f };
    entries[1]          = { 0, 1, 2, 1, 0.5f };
    in.expected_entries = 3;
    auto incomplete     = ggml_sycl::test_moe_build_same_expert_grouping(in);
    CHECK(!incomplete.accepted, "incomplete metadata must reject");
    CHECK(incomplete.reason && std::string(incomplete.reason) == "shape", "incomplete reject reason must be shape");
    return 0;
}

static int test_no_lane_filled_group_falls_back() {
    const int32_t                                               ids[]     = { 0, 1, 2, 3 };
    const float                                                 weights[] = { 0.25f, 0.25f, 0.25f, 0.25f };
    std::vector<ggml_sycl::test_moe_token_major_metadata_entry> entries;
    if (build_token_major_entries(ids, weights, 4, 1, 8, entries) != 0) {
        return 1;
    }

    ggml_sycl::test_moe_same_expert_group          groups[4]{};
    int32_t                                        grouped_indices[4]{};
    ggml_sycl::test_moe_same_expert_grouping_input in{};
    in.entries            = entries.data();
    in.entry_count        = static_cast<int64_t>(entries.size());
    in.expected_entries   = static_cast<int64_t>(entries.size());
    in.n_tokens           = 1;
    in.n_experts          = 8;
    in.max_lanes          = 8;
    in.out_groups         = groups;
    in.out_group_capacity = 4;
    in.out_entry_indices  = grouped_indices;
    in.out_entry_capacity = 4;
    auto out              = ggml_sycl::test_moe_build_same_expert_grouping(in);

    CHECK(!out.accepted, "all-singleton grouping must fail closed so runtime falls back");
    CHECK(out.reason && std::string(out.reason) == "lanes", "all-singleton reject reason must be lanes");
    CHECK(!out.has_lane_filled_group, "all-singleton grouping must report no lane-filled group");
    CHECK(out.groups == 4 && out.grouped_entries == 4, "fallback decision must still have deterministic coverage");

    ggml_sycl::test_moe_down_sum_direct_final_input policy{};
    policy.dpas_direct_final_env_enabled                 = true;
    policy.dpas_direct_final_i8_enabled                  = true;
    policy.dpas_direct_final_same_expert_grouped_enabled = true;
    policy.n_tokens                                      = 1;
    policy.down_layout                                   = GGML_LAYOUT_MXFP4_I8;
    policy.cached_q8_direct_ids                          = true;
    policy.has_ids_host                                  = true;
    policy.ids_host_count                                = 4;
    policy.down_entries                                  = 4;
    policy.has_weighted_dst                              = true;
    policy.has_moe_weights                               = true;
    policy.has_down_sum_final                            = true;
    policy.down_sum_final_index                          = 0;
    policy.glu_handle_device                             = true;
    policy.final_handle_device                           = true;
    policy.down_weight_resident                          = true;
    policy.token_major_deterministic_metadata            = true;
    policy.token_major_entries                           = 4;
    policy.token_major_expected_entries                  = 4;
    policy.token_major_weights_valid                     = true;
    policy.same_expert_grouping_metadata                 = true;
    policy.same_expert_grouping_has_lane_filled_group    = out.has_lane_filled_group;
    auto policy_out                                      = ggml_sycl::test_moe_down_sum_direct_final_policy(policy);
    CHECK(!policy_out.accepted, "policy must reject grouped variant when grouping has no lane-filled group");
    CHECK(policy_out.reason && std::string(policy_out.reason) == "same-expert-lanes",
          "policy no-lane reason must be same-expert-lanes");
    return 0;
}

static int test_runtime_hooks_are_default_off_and_labeled() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_SAME_EXPERT_GROUPED"),
          "same-expert grouped direct-final path must be explicitly env-gated");
    CHECK(contains(mmvq, "down-dpas-direct-final-same-expert-grouped"),
          "same-expert grouped direct-final profile label must be present");
    CHECK(contains(mmvq, "has_lane_filled_group"),
          "runtime grouping must fail closed when no group fills multiple DPAS lanes");
    return 0;
}

int main() {
    if (test_groups_same_expert_entries_stably() != 0) {
        return 1;
    }
    if (test_rejects_incomplete_or_unsorted_metadata() != 0) {
        return 1;
    }
    if (test_no_lane_filled_group_falls_back() != 0) {
        return 1;
    }
    if (test_runtime_hooks_are_default_off_and_labeled() != 0) {
        return 1;
    }
    std::puts("PASS: same-expert grouped direct-final grouping");
    return 0;
}
