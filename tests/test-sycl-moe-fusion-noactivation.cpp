#include <algorithm>
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

static bool contains(const std::string & haystack, const char * needle) {
    return haystack.find(needle) != std::string::npos;
}

static std::string join_path(const std::string & root, const char * rel) {
    if (root.empty() || root == ".") {
        return rel;
    }
    if (root.back() == '/') {
        return root + rel;
    }
    return root + "/" + rel;
}

static std::vector<std::string> candidate_roots() {
    std::vector<std::string> roots;
    if (const char * env = std::getenv("LLAMA_CPP_REPO_ROOT")) {
        roots.emplace_back(env);
    }
    const std::string source_file = __FILE__;
    const std::string suffix      = "/tests/test-sycl-moe-fusion-noactivation.cpp";
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
        const std::string path = join_path(root, rel);
        std::ifstream     in(path, std::ios::binary);
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

static int test_probe_symbols_exist() {
    const std::string header = read_required_file("ggml/src/ggml-sycl/ggml-sycl-test.hpp");
    const std::string sycl   = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string mmvq   = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    CHECK(contains(header, "test_moe_token_major_metadata_input"), "token-major metadata test seam must exist");
    CHECK(contains(header, "token_major_entries"), "direct-final token-major entry field must exist");
    CHECK(contains(header, "token_major_expected_entries"), "direct-final token-major expected-entry field must exist");
    CHECK(contains(header, "token_major_weights_valid"), "direct-final token-major weight-valid field must exist");
    CHECK(contains(header, "test_moe_glu_q8_artifact_input"), "GLU/Q8 artifact policy test seam must exist");
    CHECK(contains(header, "test_moe_glu_q8_counters"), "GLU/Q8 artifact counters must exist");
    CHECK(contains(sycl, "test_moe_glu_q8_artifact_policy"), "GLU/Q8 artifact policy helper must exist");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_GLU_Q8_PUBLISH_DIAG"), "GLU/Q8 diagnostic env must exist");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX"), "GLU/Q8 fused-XMX env must exist");
    return 0;
}

static int test_glu_q8_fused_xmx_env_default_off() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    const char * env_name   = "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX";
    const size_t helper_pos = mmvq.find("mxfp4_moe_glu_q8_fused_xmx_enabled");
    CHECK(helper_pos != std::string::npos, "fused-XMX env helper must exist");
    const size_t next_helper = mmvq.find("static bool mxfp4_moe_xmx_tiled_pack_q8_enabled", helper_pos);
    CHECK(next_helper != std::string::npos, "fused-XMX helper must stay near GLU/Q8 helper block");
    const std::string helper = mmvq.substr(helper_pos, next_helper - helper_pos);

    CHECK(contains(helper, env_name), "fused-XMX helper must read the expected env");
    CHECK(contains(helper, "if (!env || std::atoi(env) == 0)"),
          "fused-XMX helper must reject unset/zero env by default");
    CHECK(contains(helper, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX_UNSAFE"),
          "fused-XMX helper must require an explicit unsafe override");
    CHECK(!contains(helper, ": true"), "fused-XMX helper must not default-enable when env is unset");
    CHECK(!contains(helper, "return !(env"), "fused-XMX helper must not invert the env into default-on behavior");

    size_t env_pos = mmvq.find(env_name);
    while (env_pos != std::string::npos) {
        const size_t      begin  = env_pos > 160 ? env_pos - 160 : 0;
        const size_t      end    = std::min(mmvq.size(), env_pos + 240);
        const std::string window = mmvq.substr(begin, end - begin);
        CHECK(!contains(window, ": true"), "fused-XMX env must not appear in default-enable ternary");
        CHECK(!contains(window, "return !(env"), "fused-XMX env must not appear in inverted default-on expression");
        env_pos = mmvq.find(env_name, env_pos + 1);
    }
    return 0;
}

static int test_glu_q8_fused_xmx_runtime_quarantined() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    const char * env_name = "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX";
    CHECK(contains(mmvq, env_name), "fused-XMX env must remain discoverable");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX_UNSAFE"),
          "runtime fused-XMX must require an explicit unsafe override after B50 correctness failure");
    CHECK(contains(mmvq, "disabled-b50-incorrect") || contains(mmvq, "quarantined"),
          "runtime fused-XMX must expose a stable B50 quarantine reject reason");
    CHECK(contains(mmvq, "fused_reject"), "diagnostics must keep a stable fused_reject field");
    CHECK(contains(mmvq, "return unsafe && std::atoi(unsafe) != 0;"),
          "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX alone must not activate runtime fused store");
    return 0;
}

static int test_runtime_fusion_remains_opt_in_until_default_promotion() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");

    CHECK(contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL"),
          "runtime fusion env must remain visible for diagnostics");
    CHECK(contains(mmvq, "mxfp4_moe_down_sum_xmx_direct_final_requested"),
          "runtime fusion must use an explicit env selector helper");
    CHECK(contains(mmvq, "token_major_deterministic_metadata"),
          "runtime fusion must require deterministic token-major metadata");
    CHECK(contains(mmvq, "ggml_sycl_moe_down_sum_xmx_direct_final_mxfp4"),
          "runtime fusion selector must call the direct-final candidate only after policy acceptance");
    CHECK(contains(mmvq, "stub-fallback"),
          "unproven direct-final runtime candidate must fail closed to fallback");
    CHECK(!contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL_DEFAULT_ON"),
          "runtime fusion must not introduce a default-on env in this task");
    CHECK(contains(mmvq, "return env && std::atoi(env) != 0"),
          "direct-final selector must remain explicit opt-in by env");
    CHECK(contains(mmvq, "\"metadata\"") && contains(mmvq, "\"quarantined\"") &&
              contains(mmvq, "\"capacity\"") && contains(mmvq, "\"device\""),
          "runtime selector must expose stable rejection reasons");
    const size_t policy_pos = mmvq.find("static mxfp4_moe_direct_final_runtime_policy mxfp4_moe_down_sum_xmx_direct_final_policy");
    const size_t metadata_pos = mmvq.find("\"metadata\"", policy_pos);
    const size_t capacity_pos = mmvq.find("\"capacity\"", policy_pos);
    const size_t device_pos = mmvq.find("\"device\"", policy_pos);
    CHECK(policy_pos != std::string::npos && metadata_pos != std::string::npos && capacity_pos != std::string::npos &&
              device_pos != std::string::npos && capacity_pos < metadata_pos && device_pos < metadata_pos,
          "runtime selector must diagnose capacity/device rejects before metadata blocks make them unreachable");
    CHECK(contains(mmvq, "direct_final_layout_supported") && contains(mmvq, "direct_final_layout_probe") &&
              contains(mmvq, "down_layout == GGML_LAYOUT_XMX_TILED"),
          "runtime direct-final diagnostics must be reachable for XMX_TILED layouts before SOA fallback rejects");
    CHECK(contains(mmvq, "mxfp4_moe_down_sum_xmx_direct_final_log_reject") &&
              contains(mmvq, "direct_final_capacity_policy(false)"),
          "runtime direct-final must log stable capacity rejects before early Q8-capacity returns");
    CHECK(contains(sycl, "direct_final_probe.fused_q8_quarantined") &&
              contains(sycl, "direct_final_probe.q8_capacity_ok") && contains(sycl, "direct_final_probe.device_xmx_ok"),
          "ggml-sycl direct-final probe must preserve quarantine/capacity/device reasons before metadata");
    CHECK(contains(sycl, "ggml_sycl_moe_down_sum_cached_q8_capacity_ok"),
          "ggml-sycl direct-final probe must use the runtime cached-Q8 capacity check");
    return 0;
}

static int test_no_graphlet_or_atomic_dependency_for_probes() {
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");

    CHECK(!contains(mmvq, "MOE_DOWN_SUM_XMX_DIRECT_FINAL_GRAPHLET"), "direct-final probe must not depend on graphlets");
    CHECK(!contains(sycl, "MOE_DOWN_SUM_XMX_DIRECT_FINAL_GRAPHLET"),
          "direct-final policy probe must not depend on graphlets");
    CHECK(!contains(mmvq, "direct_final_atomic"), "direct-final probe must not introduce atomic path");
    CHECK(!contains(sycl, "direct_final_atomic"), "direct-final policy probe must not introduce atomic path");
    CHECK(!contains(mmvq, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL_ATOMIC"),
          "direct-final probe must not introduce atomic env path");
    CHECK(!contains(sycl, "GGML_SYCL_MOE_DOWN_SUM_XMX_DIRECT_FINAL_ATOMIC"),
          "direct-final policy probe must not introduce atomic env path");
    CHECK(!contains(mmvq, "mxfp4_xmx_tiled_grouped_direct_final"),
          "Task 6 may add a rejecting selector/stub but not an unreviewed direct-final XMX kernel");
    return 0;
}

static int test_promoted_default_does_not_enable_quarantined_fused_q8() {
    const std::string mmvq = read_required_file("ggml/src/ggml-sycl/mmvq.cpp");
    const std::string sycl = read_required_file("ggml/src/ggml-sycl/ggml-sycl.cpp");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_GLU_Q8_FUSED_XMX_UNSAFE"),
          "known incorrect fused-Q8 XMX path must remain behind unsafe override");
    CHECK(contains(mmvq, "disabled-b50-incorrect") || contains(mmvq, "quarantined"),
          "fused-Q8 quarantine reason must remain visible after default promotion");
    CHECK(contains(mmvq, "mxfp4_moe_down_sum_xmx_direct_final_kernel_proven") &&
              contains(mmvq, "return false;"),
          "default fast path must not enable unproven direct-final fusion");
    CHECK(contains(mmvq, "GGML_SYCL_MOE_DEFAULT_FAST_PATH") && contains(mmvq, "promotion-candidate-only"),
          "fusion selector must document default promotion as candidate-only until proven");
    CHECK(contains(mmvq, "fusion-exception") && contains(sycl, "replay-exception"),
          "fatal fusion/replay exceptions must have stable quarantine reasons");
    return 0;
}

int main() {
    if (test_probe_symbols_exist() != 0) {
        return 1;
    }
    if (test_glu_q8_fused_xmx_env_default_off() != 0) {
        return 1;
    }
    if (test_glu_q8_fused_xmx_runtime_quarantined() != 0) {
        return 1;
    }
    if (test_runtime_fusion_remains_opt_in_until_default_promotion() != 0) {
        return 1;
    }
    if (test_no_graphlet_or_atomic_dependency_for_probes() != 0) {
        return 1;
    }
    if (test_promoted_default_does_not_enable_quarantined_fused_q8() != 0) {
        return 1;
    }
    std::puts("PASS: MoE fusion probes no-activation guard");
    return 0;
}
