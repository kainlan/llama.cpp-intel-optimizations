#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/mmvq.hpp"

#include <cstdio>
#include <cstring>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static ggml_sycl::test_moe_gateup_prepack_policy_input base_input() {
    ggml_sycl::test_moe_gateup_prepack_policy_input in{};
    in.prepack_env            = "1";
    in.ne12                   = 1;
    in.layer                  = 7;
    in.selected_entries       = 4;
    in.selected_batches       = 1;
    in.metadata_complete      = true;
    in.metadata_deterministic = true;
    in.gate_handle_valid      = true;
    in.gate_handle_device     = true;
    in.up_handle_valid        = true;
    in.up_handle_device       = true;
    in.scratch_handle_valid   = true;
    in.scratch_required_bytes = 2048;
    in.scratch_capacity_bytes = 4096;
    in.graph_recording        = false;
    return in;
}

static int test_env_default_off() {
    CHECK(std::strcmp(ggml_sycl::test_moe_gateup_prepack_env_name(), "GGML_SYCL_MOE_GATEUP_PREPACK") == 0,
          "policy helper must name the opt-in env");
    CHECK(!ggml_sycl::test_moe_gateup_prepack_enabled_from_env(nullptr), "null env must be default-off");
    CHECK(!ggml_sycl::test_moe_gateup_prepack_enabled_from_env("0"), "zero env must be disabled");
    CHECK(ggml_sycl::test_moe_gateup_prepack_enabled_from_env("1"), "non-zero env must enable policy");

    auto in          = base_input();
    in.prepack_env   = nullptr;
    auto default_off = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!default_off.accepted, "unset env must reject");
    CHECK(std::strcmp(default_off.reason, "env") == 0, "unset env reject reason must be env");

    in             = base_input();
    in.prepack_env = "0";
    auto env_zero  = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!env_zero.accepted, "zero env must reject");
    CHECK(std::strcmp(env_zero.reason, "env") == 0, "zero env reject reason must be env");
    return 0;
}

static int test_rejects_malformed_decode_shape() {
    auto in   = base_input();
    in.ne12   = 2;
    auto prom = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!prom.accepted, "prompt/non-TG ne12 must reject");
    CHECK(std::strcmp(prom.reason, "shape") == 0, "prompt reject reason must be shape");

    in       = base_input();
    in.layer = -1;
    auto bad_layer = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!bad_layer.accepted && std::strcmp(bad_layer.reason, "shape") == 0, "invalid layer must reject");

    in                    = base_input();
    in.selected_entries   = 0;
    auto no_selected_rows = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!no_selected_rows.accepted && std::strcmp(no_selected_rows.reason, "shape") == 0,
          "zero selected entries must reject");

    in                     = base_input();
    in.selected_batches    = 0;
    auto no_selected_batch = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!no_selected_batch.accepted && std::strcmp(no_selected_batch.reason, "shape") == 0,
          "zero selected batches must reject");
    return 0;
}

static int test_rejects_metadata_handle_capacity_and_graph_recording() {
    auto in                  = base_input();
    in.metadata_complete     = false;
    auto missing_metadata    = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!missing_metadata.accepted, "missing metadata must reject");
    CHECK(std::strcmp(missing_metadata.reason, "metadata") == 0, "metadata reject reason must be metadata");

    in                         = base_input();
    in.metadata_deterministic  = false;
    auto nondeterministic_meta = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!nondeterministic_meta.accepted && std::strcmp(nondeterministic_meta.reason, "metadata") == 0,
          "non-deterministic metadata must reject");

    in                    = base_input();
    in.gate_handle_valid  = false;
    auto missing_gate     = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!missing_gate.accepted, "missing gate handle must reject");
    CHECK(std::strcmp(missing_gate.reason, "handle") == 0, "missing gate handle reject reason must be handle");

    in                   = base_input();
    in.up_handle_device  = false;
    auto host_up         = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!host_up.accepted && std::strcmp(host_up.reason, "handle") == 0,
          "host/off-device up handle must reject");

    in                         = base_input();
    in.scratch_handle_valid    = false;
    auto missing_scratch       = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!missing_scratch.accepted && std::strcmp(missing_scratch.reason, "handle") == 0,
          "missing scratch handle must reject");

    in                         = base_input();
    in.scratch_capacity_bytes  = in.scratch_required_bytes - 1;
    auto insufficient_capacity = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!insufficient_capacity.accepted, "insufficient scratch capacity must reject");
    CHECK(std::strcmp(insufficient_capacity.reason, "capacity") == 0, "capacity reject reason must be capacity");

    in                         = base_input();
    in.scratch_required_bytes  = 0;
    auto zero_required         = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!zero_required.accepted && std::strcmp(zero_required.reason, "capacity") == 0,
          "zero scratch requirement must reject as capacity");

    in                  = base_input();
    in.graph_recording  = true;
    auto graph_recorded = ggml_sycl::test_moe_gateup_prepack_policy(in);
    CHECK(!graph_recorded.accepted, "graph recording must reject for now");
    CHECK(std::strcmp(graph_recorded.reason, "graph-recording") == 0,
          "graph recording reject reason must be graph-recording");
    return 0;
}

static ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy_input base_runtime_input() {
    ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy_input in{};
    in.env                          = "1";
    in.ne12                         = 1;
    in.layer                        = 7;
    in.selected_entries             = 4;
    in.selected_batches             = 4;
    in.metadata_complete            = true;
    in.metadata_deterministic       = true;
    in.gate_handle_valid            = true;
    in.gate_handle_device           = true;
    in.up_handle_valid              = true;
    in.up_handle_device             = true;
    in.source_handle_valid          = true;
    in.source_handle_device         = true;
    in.route_metadata_handle_valid  = true;
    in.route_metadata_handle_device = true;
    in.scratch_handle_valid         = true;
    in.scratch_handle_device        = true;
    in.scratch_required_bytes       = 2048;
    in.scratch_capacity_bytes       = 4096;
    in.graph_recording              = false;
    in.direct_down_sum_compatible   = true;
    in.prepack_supported            = true;
    in.compute_supported            = true;
    return in;
}

static int test_runtime_policy_contract() {
    CHECK(std::strcmp(ggml_sycl::mxfp4_moe_gateup_prepack_route_label(), "gateup-prepack-dpas") == 0,
          "runtime route label must identify gate/up prepack DPAS path");

    auto in = base_runtime_input();
    in.env  = nullptr;
    auto out = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(in);
    CHECK(!out.accepted, "runtime policy must preserve default-off fallback");
    CHECK(std::strcmp(out.reason, "env") == 0, "runtime default-off reason must be env");
    CHECK(std::strcmp(out.selected_route, "fallback") == 0, "env-off must not select a new route");

    out = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(base_runtime_input());
    CHECK(out.accepted, "runtime policy must accept valid opt-in input");
    CHECK(std::strcmp(out.selected_route, "gateup-prepack-dpas") == 0,
          "accepted runtime route label must be gateup-prepack-dpas");

    in                              = base_runtime_input();
    in.direct_down_sum_compatible  = false;
    out                             = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "down-sum") == 0,
          "missing direct down-sum compatibility must reject before route selection");

    in                    = base_runtime_input();
    in.prepack_supported  = false;
    out                   = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "prepack") == 0,
          "prepack failure must reject before publishing route artifacts");

    in                   = base_runtime_input();
    in.compute_supported = false;
    out                  = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "compute") == 0,
          "compute submission preflight failure must reject before route selection");

    in                             = base_runtime_input();
    in.route_metadata_handle_valid = false;
    out                            = ggml_sycl::mxfp4_moe_gateup_prepack_runtime_policy(in);
    CHECK(!out.accepted && std::strcmp(out.reason, "handle") == 0,
          "missing route metadata handle must reject as handle failure");
    return 0;
}

static int test_accepts_valid_enabled_case_and_counts() {
    ggml_sycl::test_moe_gateup_prepack_policy_counters_reset();
    auto out = ggml_sycl::test_moe_gateup_prepack_policy(base_input());
    CHECK(out.accepted, "valid opt-in gate/up prepack policy must accept");
    CHECK(std::strcmp(out.reason, "none") == 0, "accepted reason must be none");
    CHECK(std::strcmp(out.selected_route, "gateup-prepack-dpas") == 0, "accepted route label must be stable");

    auto in       = base_input();
    in.prepack_env = "0";
    (void) ggml_sycl::test_moe_gateup_prepack_policy(in);

    const auto counters = ggml_sycl::test_moe_gateup_prepack_policy_counters_snapshot();
    CHECK(counters.candidates == 2, "counters must record all candidates");
    CHECK(counters.accepted == 1, "counters must record accepted candidate");
    CHECK(counters.rejected == 1, "counters must record rejected candidate");
    return 0;
}

int main() {
    if (test_env_default_off() != 0) {
        return 1;
    }
    if (test_rejects_malformed_decode_shape() != 0) {
        return 1;
    }
    if (test_rejects_metadata_handle_capacity_and_graph_recording() != 0) {
        return 1;
    }
    if (test_accepts_valid_enabled_case_and_counts() != 0) {
        return 1;
    }
    if (test_runtime_policy_contract() != 0) {
        return 1;
    }
    std::puts("PASS: MoE gate/up prepack policy");
    return 0;
}
