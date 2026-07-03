#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/mmvq.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

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
    const std::string suffix      = "/tests/test-sycl-moe-fused-down-sum-policy.cpp";
    const size_t      pos         = source_file.rfind(suffix);
    if (pos != std::string::npos) {
        roots.emplace_back(source_file.substr(0, pos));
    }
    roots.emplace_back(".");
    roots.emplace_back("..");
    roots.emplace_back("../..");
    roots.emplace_back("../../..");
    roots.emplace_back("../../../..");
    roots.emplace_back("../../../../..");
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

static ggml_sycl::test_moe_down_sum_direct_final_input base_input() {
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
    in.down_sum_final_index               = 123;
    in.glu_handle_device                  = true;
    in.final_handle_device                = true;
    in.down_weight_resident               = true;
    in.token_major_deterministic_metadata = true;
    in.token_major_entries                = 2;
    in.token_major_expected_entries       = 2;
    in.token_major_weights_valid          = true;
    return in;
}

static int test_rejects_current_grouped_metadata_blocker() {
    auto in                               = base_input();
    in.token_major_deterministic_metadata = false;
    auto out                              = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "current grouped-by-expert metadata must reject deterministic direct-final");
    CHECK(std::strcmp(out.reason, "metadata") == 0, "current grouped metadata reject reason must be metadata");
    return 0;
}

static int test_default_disabled() {
    auto in        = base_input();
    in.env_enabled = false;
    auto out       = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "default must reject when env is disabled");
    CHECK(std::strcmp(out.reason, "env") == 0, "default reject reason must be env");
    return 0;
}

static int test_accepts_tg_xmx_weighted_final() {
    auto out = ggml_sycl::test_moe_down_sum_direct_final_policy(base_input());
    CHECK(out.accepted, "valid TG XMX weighted final path must be accepted");
    CHECK(out.expected_saved_launches == 1, "direct-final should save the reduce launch");
    return 0;
}

static int test_accepts_i8_and_dpas_direct_final_when_explicitly_enabled() {
    // This is a policy-attestation fixture: runtime I8/DPAS direct-final remains
    // metadata-blocked until the token-major bridge task supplies complete metadata.
    auto in                          = base_input();
    in.down_layout                   = GGML_LAYOUT_MXFP4_I8;
    in.dpas_direct_final_env_enabled = true;
    in.dpas_direct_final_i8_enabled  = true;
    auto i8                          = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(i8.accepted, "I8 direct-final must accept only behind explicit env");

    in                                = base_input();
    in.down_layout                    = GGML_LAYOUT_MXFP4_DPAS;
    in.dpas_direct_final_env_enabled  = true;
    in.dpas_direct_final_dpas_enabled = true;
    auto dpas                         = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(dpas.accepted, "DPAS direct-final must accept only behind explicit env");
    return 0;
}

static int test_rejects_i8_and_dpas_when_per_layout_env_missing() {
    auto in                          = base_input();
    in.down_layout                   = GGML_LAYOUT_MXFP4_I8;
    in.dpas_direct_final_env_enabled = true;
    auto i8                          = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!i8.accepted && std::strcmp(i8.reason, "env") == 0, "I8 direct-final must reject without I8 env");

    in                               = base_input();
    in.down_layout                   = GGML_LAYOUT_MXFP4_DPAS;
    in.dpas_direct_final_env_enabled = true;
    auto dpas                        = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!dpas.accepted && std::strcmp(dpas.reason, "env") == 0, "DPAS direct-final must reject without DPAS env");
    return 0;
}

static int test_rejects_prompt_and_non_xmx_layouts() {
    auto in     = base_input();
    in.n_tokens = 8;
    auto prompt = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!prompt.accepted && std::strcmp(prompt.reason, "tokens") == 0, "prompt must reject");

    in             = base_input();
    in.down_layout = GGML_LAYOUT_SOA;
    auto soa       = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!soa.accepted && std::strcmp(soa.reason, "layout") == 0, "SOA must reject Stage A XMX path");
    return 0;
}

static int test_rejects_missing_final_and_weights() {
    auto in               = base_input();
    in.has_down_sum_final = false;
    auto no_final         = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!no_final.accepted && std::strcmp(no_final.reason, "final") == 0, "missing final must reject");

    in                 = base_input();
    in.has_moe_weights = false;
    auto no_weights    = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!no_weights.accepted && std::strcmp(no_weights.reason, "weights") == 0, "missing weights must reject");
    return 0;
}

static int test_rejects_bad_ids_and_residency() {
    auto in         = base_input();
    in.has_ids_host = false;
    auto no_ids     = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!no_ids.accepted && std::strcmp(no_ids.reason, "ids") == 0, "XMX path requires host ids for grouping");

    in                     = base_input();
    in.final_handle_device = false;
    auto host_final        = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!host_final.accepted && std::strcmp(host_final.reason, "residency") == 0,
          "host/off-device final handle must reject");
    return 0;
}

static int test_rejects_quarantine_device_capacity_before_metadata() {
    auto in                               = base_input();
    in.token_major_deterministic_metadata = false;
    in.fused_q8_quarantined               = true;
    auto out                              = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "quarantined") == 0,
          "quarantined fused-Q8 must be diagnosed before metadata");

    in                     = base_input();
    in.token_major_entries = 0;
    in.device_xmx_ok       = false;
    out                    = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "device") == 0,
          "unsupported XMX device must be diagnosed before metadata");

    in                     = base_input();
    in.token_major_entries = 0;
    in.q8_capacity_ok      = false;
    out                    = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "capacity") == 0,
          "missing cached-Q8 capacity must be diagnosed before metadata");
    return 0;
}

static int test_direct_final_policy_uses_only_dpas_atomic_variant_name() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    CHECK(!contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL_ATOMIC"),
          "XMX direct-final policy must not add the old atomic shortcut env");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL_RANK_PARALLEL_ATOMIC"),
          "rank-parallel atomic must be gated by the DPAS direct-final env namespace");
    CHECK(contains(mmvq, "down-dpas-direct-final-rank-parallel-atomic"),
          "rank-parallel atomic profile label must be present");
    return 0;
}

static int test_soa_tg_row_group_variant_parser_and_labels() {
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env(nullptr) == 1,
          "missing SOA TG row-group env must default to scalar rows");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("") == 1,
          "empty SOA TG row-group env must default to scalar rows");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("0") == 1,
          "zero SOA TG row-group env must fail closed");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("bogus") == 1,
          "unknown SOA TG row-group env must fail closed");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("direct-final") == 1,
          "direct-final must not be accepted as a SOA TG row-group alias");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("2") == 1,
          "numeric row2 SOA TG row-group alias must fail closed");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("row2") == 2,
          "named row2 SOA TG row-group env must parse");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("4") == 1,
          "numeric row4 SOA TG row-group alias must fail closed");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_rows_per_group_from_env("row4") == 4,
          "named row4 SOA TG row-group env must parse");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_variant_label(1) == nullptr,
          "default scalar-row SOA TG route must not emit an opt-in path label");
    CHECK(std::strcmp(ggml_sycl_moe_down_sum_q8_soa_tg_variant_label(2), "down-soa-q8-tg-row2") == 0,
          "row2 SOA TG variant label must be stable");
    CHECK(std::strcmp(ggml_sycl_moe_down_sum_q8_soa_tg_variant_label(4), "down-soa-q8-tg-row4") == 0,
          "row4 SOA TG variant label must be stable");

    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env("row2", false, 1) == 1,
          "row2 SOA TG row-group env must not affect non-DOWN roles");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env("row4", true, 2) == 1,
          "row4 SOA TG row-group env must not affect prompt/multi-token batches");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env("row2", true, 1) == 2,
          "row2 SOA TG row-group env must be active for DOWN decode only");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env("row4", true, 1) == 4,
          "row4 SOA TG row-group env must be active for DOWN decode only");
    CHECK(ggml_sycl_moe_down_sum_q8_soa_tg_active_rows_per_group_from_env("4", true, 1) == 1,
          "numeric aliases must still fail closed in the active DOWN decode policy");
    return 0;
}

static int test_cached_down_q8_soa_tg_variant_parser_labels_and_scope() {
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env(nullptr) == 0,
          "missing cached-Q8 SOA TG variant env must default off");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("") == 0,
          "empty cached-Q8 SOA TG variant env must default off");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("0") == 0,
          "zero cached-Q8 SOA TG variant env must fail closed");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("1") == 0,
          "numeric cached-Q8 SOA TG variant aliases must fail closed");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("bogus") == 0,
          "unknown cached-Q8 SOA TG variant env must fail closed");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("direct-final") == 0,
          "direct-final must not be accepted as a cached-Q8 SOA TG variant alias");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("vector-qs") == 1,
          "vector-qs cached-Q8 SOA TG variant must parse");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("cache-y") == 2,
          "cache-y cached-Q8 SOA TG variant must parse");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_from_env("vector-qs-cache-y") == 3,
          "combined cached-Q8 SOA TG variant must parse");

    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_active_variant_from_env("vector-qs", false, true, 1) == 0,
          "cached-Q8 SOA TG variant must not affect non-DOWN roles");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_active_variant_from_env("vector-qs", true, false, 1) == 0,
          "cached-Q8 SOA TG variant must not affect non-SOA layouts");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_active_variant_from_env("vector-qs", true, true, 2) == 0,
          "cached-Q8 SOA TG variant must not affect prompt/multi-token batches");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_active_variant_from_env("vector-qs", true, true, 1) == 1,
          "cached-Q8 SOA TG variant must be active for SOA DOWN decode only");

    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_label(0) == nullptr,
          "default cached-Q8 SOA TG route must not emit an opt-in path label");
    CHECK(std::strcmp(ggml_sycl_moe_down_cached_q8_soa_tg_variant_label(1), "down-soa-cached-q8-tg-vector-qs") == 0,
          "vector-qs cached-Q8 SOA TG variant label must be stable");
    CHECK(std::strcmp(ggml_sycl_moe_down_cached_q8_soa_tg_variant_label(2), "down-soa-cached-q8-tg-cache-y") == 0,
          "cache-y cached-Q8 SOA TG variant label must be stable");
    CHECK(std::strcmp(ggml_sycl_moe_down_cached_q8_soa_tg_variant_label(3),
                      "down-soa-cached-q8-tg-vector-qs-cache-y") == 0,
          "combined cached-Q8 SOA TG variant label must be stable");

    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_vector_qs_load(1),
          "vector-qs variant must enable vector_qs_load");
    CHECK(!ggml_sycl_moe_down_cached_q8_soa_tg_variant_cache_y_local(1),
          "vector-qs variant must not enable cache_y_local");
    CHECK(!ggml_sycl_moe_down_cached_q8_soa_tg_variant_vector_qs_load(2),
          "cache-y variant must not enable vector_qs_load");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_cache_y_local(2), "cache-y variant must enable cache_y_local");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_vector_qs_load(3), "combined variant must enable vector_qs_load");
    CHECK(ggml_sycl_moe_down_cached_q8_soa_tg_variant_cache_y_local(3), "combined variant must enable cache_y_local");

    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT"),
          "cached-Q8 SOA TG variant must be gated by its explicit env");
    CHECK(contains(mmvq, "row_group_variant == 1 ?"),
          "cached-Q8 SOA TG variants must not override explicit row2/row4 row-group variants");
    CHECK(contains(mmvq, "down-soa-cached-q8-tg-vector-qs-cache-y"),
          "combined cached-Q8 SOA TG profile label must be emitted only by the cached DOWN branch");
    return 0;
}

static int test_down_q8_soa_variants_do_not_enable_direct_final() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL"),
          "direct-final env knob must remain explicit");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_DPAS_DIRECT_FINAL"),
          "DPAS direct-final env knob must remain explicit");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT"),
          "q8-SOA row-group env must remain separate from direct-final envs");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT"),
          "cached q8-SOA env must remain separate from direct-final envs");
    CHECK(!contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_Q8_SOA_TG_VARIANT=direct-final"),
          "row-group env must not grow a direct-final string alias");
    CHECK(!contains(mmvq, "GGML_SYCL_MOE_DOWN_CACHED_Q8_SOA_TG_VARIANT=direct-final"),
          "cached q8-SOA env must not grow a direct-final string alias");
    return 0;
}

static int test_dpas_concurrency_variants_are_explicit_and_exclusive() {
    auto in                          = base_input();
    in.down_layout                   = GGML_LAYOUT_MXFP4_I8;
    in.env_enabled                   = false;
    in.dpas_direct_final_env_enabled = true;
    in.dpas_direct_final_i8_enabled  = true;

    auto baseline = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(baseline.accepted, "existing I8 direct-final must still accept without a concurrency variant");
    CHECK(std::strcmp(baseline.selected_variant, "serial") == 0, "baseline selected variant must be serial");

    in.dpas_direct_final_rank_parallel_atomic_enabled = true;
    auto atomic                                       = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(atomic.accepted, "rank-parallel atomic variant must accept behind explicit env");
    CHECK(std::strcmp(atomic.selected_variant, "rank-parallel-atomic") == 0,
          "atomic selected variant must be rank-parallel-atomic");

    in.dpas_direct_final_scratch_reduce_enabled = true;
    auto conflict                               = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!conflict.accepted, "multiple concurrency variants must reject");
    CHECK(std::strcmp(conflict.reason, "variant") == 0, "variant conflict reject reason must be variant");

    in.dpas_direct_final_rank_parallel_atomic_enabled = false;
    in.dpas_direct_final_scratch_reduce_enabled       = true;
    in.dpas_direct_final_same_expert_grouped_enabled  = false;
    auto scratch                                      = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(scratch.accepted, "scratch+reduce variant must accept behind explicit env");
    CHECK(std::strcmp(scratch.selected_variant, "scratch-reduce") == 0,
          "scratch selected variant must be scratch-reduce");

    in.dpas_direct_final_scratch_reduce_enabled      = false;
    in.dpas_direct_final_same_expert_grouped_enabled = true;
    auto missing_grouping                            = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!missing_grouping.accepted, "same-expert grouped variant must fail closed without grouping metadata");
    CHECK(std::strcmp(missing_grouping.reason, "same-expert-metadata") == 0,
          "missing grouped metadata reason must be same-expert-metadata");

    in.same_expert_grouping_metadata = true;
    auto no_lane_group               = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!no_lane_group.accepted, "same-expert grouped variant must fallback when no group fills multiple lanes");
    CHECK(std::strcmp(no_lane_group.reason, "same-expert-lanes") == 0,
          "no-lane grouped metadata reason must be same-expert-lanes");

    in.same_expert_grouping_has_lane_filled_group = true;
    auto grouped                                  = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(grouped.accepted, "same-expert grouped variant must accept behind explicit env with lane-filled metadata");
    CHECK(std::strcmp(grouped.selected_variant, "same-expert-grouped") == 0,
          "grouped selected variant must be same-expert-grouped");
    return 0;
}

static int test_dpas_concurrency_variants_do_not_bypass_direct_final_umbrella() {
    auto in                                           = base_input();
    in.down_layout                                    = GGML_LAYOUT_MXFP4_DPAS;
    in.env_enabled                                    = false;
    in.dpas_direct_final_env_enabled                  = false;
    in.dpas_direct_final_dpas_enabled                 = true;
    in.dpas_direct_final_rank_parallel_atomic_enabled = true;
    auto out                                          = ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    CHECK(!out.accepted, "variant env must not bypass direct-final umbrella");
    CHECK(std::strcmp(out.reason, "env") == 0, "missing umbrella reject reason must be env");
    return 0;
}

static int test_counters() {
    ggml_sycl::test_moe_down_sum_direct_final_counters_reset();
    auto in = base_input();
    (void) ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    in.n_tokens = 4;
    (void) ggml_sycl::test_moe_down_sum_direct_final_policy(in);
    ggml_sycl::test_moe_down_sum_direct_final_record_submit_saved_for_test();

    auto snap = ggml_sycl::test_moe_down_sum_direct_final_counters_snapshot();
    CHECK(snap.candidates == 2, "candidate counter must count all policy checks");
    CHECK(snap.accepted == 1, "accepted counter must count accepted policy checks");
    CHECK(snap.rejected == 1, "rejected counter must count rejected policy checks");
    CHECK(snap.submit_saved == 1, "submit-saved counter must be recordable");
    return 0;
}

int main() {
    if (test_rejects_current_grouped_metadata_blocker() != 0) {
        return 1;
    }
    if (test_default_disabled() != 0) {
        return 1;
    }
    if (test_accepts_tg_xmx_weighted_final() != 0) {
        return 1;
    }
    if (test_accepts_i8_and_dpas_direct_final_when_explicitly_enabled() != 0) {
        return 1;
    }
    if (test_rejects_i8_and_dpas_when_per_layout_env_missing() != 0) {
        return 1;
    }
    if (test_rejects_prompt_and_non_xmx_layouts() != 0) {
        return 1;
    }
    if (test_rejects_missing_final_and_weights() != 0) {
        return 1;
    }
    if (test_rejects_bad_ids_and_residency() != 0) {
        return 1;
    }
    if (test_rejects_quarantine_device_capacity_before_metadata() != 0) {
        return 1;
    }
    if (test_direct_final_policy_uses_only_dpas_atomic_variant_name() != 0) {
        return 1;
    }
    if (test_soa_tg_row_group_variant_parser_and_labels() != 0) {
        return 1;
    }
    if (test_cached_down_q8_soa_tg_variant_parser_labels_and_scope() != 0) {
        return 1;
    }
    if (test_down_q8_soa_variants_do_not_enable_direct_final() != 0) {
        return 1;
    }
    if (test_dpas_concurrency_variants_are_explicit_and_exclusive() != 0) {
        return 1;
    }
    if (test_dpas_concurrency_variants_do_not_bypass_direct_final_umbrella() != 0) {
        return 1;
    }
    if (test_counters() != 0) {
        return 1;
    }
    std::puts("PASS: MoE XMX down-sum direct-final policy");
    return 0;
}
