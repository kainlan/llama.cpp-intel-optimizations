#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static ggml_sycl::test_moe_glu_q8_fused_store_input base_input() {
    ggml_sycl::test_moe_glu_q8_fused_store_input in{};
    in.env_enabled       = true;
    in.artifact_enabled  = true;
    in.n_tokens          = 1;
    in.glu_row_contig    = true;
    in.glu_handle_valid  = true;
    in.glu_handle_device = true;
    in.kernel_event_set  = true;
    in.q8_capacity_bytes = 4096;
    in.q8_required_bytes = 2048;
    in.kernel_path       = ggml_sycl::test_moe_glu_q8_kernel_path::SPLIT_SG16;
    in.kernel_writes_q8  = true;
    return in;
}

static int test_accepts_supported_fused_store() {
    ggml_sycl::test_moe_glu_q8_fused_store_counters_reset();
    auto out = ggml_sycl::test_moe_glu_q8_fused_store_policy(base_input());
    CHECK(out.accepted, "supported fused-store path must accept");
    CHECK(std::strcmp(out.reason, "none") == 0, "accepted reason must be none");
    CHECK(out.expected_saved_launches == 1, "fused-store saves one publish launch");
    auto counters = ggml_sycl::test_moe_glu_q8_fused_store_counters_snapshot();
    CHECK(counters.candidates == 1 && counters.accepted == 1 && counters.rejected == 0,
          "counters must record accepted candidate");
    return 0;
}

static int test_rejects_unsupported_grouped_path_until_kernel_support_exists() {
    auto in             = base_input();
    in.kernel_path      = ggml_sycl::test_moe_glu_q8_kernel_path::GROUPED_DIRECT_Q8;
    in.kernel_writes_q8 = false;
    auto out            = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!out.accepted, "grouped path without q8 writer must reject");
    CHECK(std::strcmp(out.reason, "kernel") == 0, "unsupported kernel reject reason must be kernel");
    return 0;
}

static int test_rejects_env_shape_handle_capacity_and_event() {
    auto in        = base_input();
    in.env_enabled = false;
    auto no_env    = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!no_env.accepted && std::strcmp(no_env.reason, "env") == 0, "disabled env must reject");

    in                  = base_input();
    in.artifact_enabled = false;
    auto no_artifact    = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!no_artifact.accepted && std::strcmp(no_artifact.reason, "env") == 0,
          "disabled q8 artifact must reject as env");

    in          = base_input();
    in.n_tokens = 4;
    auto prompt = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!prompt.accepted && std::strcmp(prompt.reason, "shape") == 0, "prompt must reject as shape");

    in                = base_input();
    in.glu_row_contig = false;
    auto non_contig   = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!non_contig.accepted && std::strcmp(non_contig.reason, "shape") == 0,
          "non-contiguous GLU rows must reject as shape");

    in                  = base_input();
    in.glu_handle_valid = false;
    auto no_handle      = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!no_handle.accepted && std::strcmp(no_handle.reason, "handle") == 0, "missing handle must reject");

    in                   = base_input();
    in.glu_handle_device = false;
    auto host            = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!host.accepted && std::strcmp(host.reason, "handle") == 0, "host handle must reject");

    in                   = base_input();
    in.q8_capacity_bytes = 1024;
    auto cap             = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!cap.accepted && std::strcmp(cap.reason, "capacity") == 0, "small q8 capacity must reject");

    in                    = base_input();
    in.q8_required_bytes  = 0;
    auto no_capacity_need = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!no_capacity_need.accepted && std::strcmp(no_capacity_need.reason, "capacity") == 0,
          "zero q8 requirement must reject as capacity");

    in                  = base_input();
    in.kernel_event_set = false;
    auto event          = ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    CHECK(!event.accepted && std::strcmp(event.reason, "event") == 0, "missing event must reject");
    return 0;
}

static int test_counters_record_rejections() {
    ggml_sycl::test_moe_glu_q8_fused_store_counters_reset();
    auto in     = base_input();
    in.n_tokens = 2;
    (void) ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    in             = base_input();
    in.kernel_path = ggml_sycl::test_moe_glu_q8_kernel_path::NONE;
    (void) ggml_sycl::test_moe_glu_q8_fused_store_policy(in);
    auto counters = ggml_sycl::test_moe_glu_q8_fused_store_counters_snapshot();
    CHECK(counters.candidates == 2 && counters.accepted == 0 && counters.rejected == 2,
          "counters must record rejected candidates");
    return 0;
}

int main() {
    if (test_accepts_supported_fused_store() != 0) {
        return 1;
    }
    if (test_rejects_unsupported_grouped_path_until_kernel_support_exists() != 0) {
        return 1;
    }
    if (test_rejects_env_shape_handle_capacity_and_event() != 0) {
        return 1;
    }
    if (test_counters_record_rejections() != 0) {
        return 1;
    }
    std::puts("PASS: MoE GLU/Q8 fused-store policy");
    return 0;
}
