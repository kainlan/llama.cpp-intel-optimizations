#include "ggml-sycl/sycl-kernel-profiler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

static void set_env_var(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void clear_env_var(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

static void clear_profile_env() {
    clear_env_var("GGML_SYCL_KERNEL_PROFILE");
    clear_env_var("GGML_SYCL_KERNEL_PROFILE_OUTPUT");
    clear_env_var("GGML_SYCL_KERNEL_PROFILE_FORMAT");
    clear_env_var("GGML_SYCL_KERNEL_PROFILE_TOP_N");
    clear_env_var("GGML_SYCL_KERNEL_PROFILE_RAW");
    clear_env_var("GGML_SYCL_KERNEL_PROFILE_FLUSH");
}

int main() {
    ggml_sycl_kernel_profile_reset_for_test();
    clear_profile_env();

    const ggml_sycl_kernel_profile_config disabled_env = ggml_sycl_kernel_profile_config_from_env();
    CHECK(!disabled_env.enabled, "unset GGML_SYCL_KERNEL_PROFILE should disable profiler");

    set_env_var("GGML_SYCL_KERNEL_PROFILE", "1");
    set_env_var("GGML_SYCL_KERNEL_PROFILE_OUTPUT", "/tmp/sycl-kernels");
    set_env_var("GGML_SYCL_KERNEL_PROFILE_FORMAT", "both");
    set_env_var("GGML_SYCL_KERNEL_PROFILE_TOP_N", "7");
    set_env_var("GGML_SYCL_KERNEL_PROFILE_RAW", "1");
    set_env_var("GGML_SYCL_KERNEL_PROFILE_FLUSH", "window");

    const ggml_sycl_kernel_profile_config parsed_env = ggml_sycl_kernel_profile_config_from_env();
    CHECK(parsed_env.enabled, "enabled env did not enable profiler");
    CHECK(parsed_env.output_path == "/tmp/sycl-kernels", "output path env not parsed");
    CHECK(parsed_env.output_format == ggml_sycl_kernel_profile_output_format::BOTH, "format env not parsed");
    CHECK(parsed_env.top_n == 7, "top-N env not parsed");
    CHECK(parsed_env.raw_events, "raw-events env not parsed");
    CHECK(parsed_env.flush_mode == ggml_sycl_kernel_profile_flush_mode::WINDOW, "flush mode env not parsed");
    clear_profile_env();

    ggml_sycl_kernel_profile_config cfg{};
    cfg.enabled       = true;
    cfg.output_format = ggml_sycl_kernel_profile_output_format::BOTH;
    cfg.flush_mode    = ggml_sycl_kernel_profile_flush_mode::WINDOW;
    cfg.top_n         = 2;
    cfg.raw_events    = true;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);

    ggml_sycl_profile_label slow{};
    slow.name       = "mxfp4.gateup.packed_q8_m2";
    slow.category   = "mmvq";
    slow.queue_kind = "compute";
    slow.metadata   = "shape=m2880n4k2880,path=packed-q8-m2";
    slow.device     = 1;
    slow.bytes      = 4096;

    ggml_sycl_profile_label fast{};
    fast.name       = "sycl.memcpy.graph_safe";
    fast.category   = "memory";
    fast.queue_kind = "compute";
    fast.metadata   = "bytes=1024";
    fast.device     = 1;
    fast.bytes      = 1024;

    ggml_sycl_kernel_profile_add_sample_for_test(slow, 1000);
    ggml_sycl_kernel_profile_add_sample_for_test(slow, 3000);
    ggml_sycl_kernel_profile_add_sample_for_test(fast, 500);
    ggml_sycl_kernel_profile_add_failed_timestamp_for_test(fast, false);
    ggml_sycl_kernel_profile_add_failed_timestamp_for_test(fast, true);

    const std::string csv = ggml_sycl_kernel_profile_format_csv_for_test();
    CHECK(contains(csv,
                   "name,category,metadata,device,queue_kind,count,total_ns,mean_ns,min_ns,p50_ns,p95_ns,max_ns,bytes,"
                   "failed_timestamps,graph_recorded"),
          "missing CSV header");
    CHECK(contains(csv,
                   "mxfp4.gateup.packed_q8_m2,mmvq,shape=m2880n4k2880;path=packed-q8-m2,1,compute,2,4000,2000,1000,"
                   "1000,1000,3000,8192,0,0"),
          "missing slow aggregate row");
    CHECK(contains(csv, "sycl.memcpy.graph_safe,memory,bytes=1024,1,compute,1,500,500,500,500,500,500,1024,2,1"),
          "missing memcpy aggregate row");

    ggml_sycl_kernel_profile_add_raw_event_for_test(slow, 42, 100, 110, 1000, 1200, 1800, "test",
                                                    "tests/test-sycl-kernel-profiler.cpp", 123, "main", true);

    const std::string json = ggml_sycl_kernel_profile_format_json_for_test();
    CHECK(contains(json, "\"kernels\""), "missing kernels JSON array");
    CHECK(contains(json, "\"name\":\"mxfp4.gateup.packed_q8_m2\""), "missing slow JSON name");
    CHECK(contains(json, "\"total_ns\":4000"), "missing slow JSON total");
    CHECK(contains(json, "\"failed_timestamps\":2"), "missing failed timestamp count");
    CHECK(contains(json, "\"raw_events\""), "missing raw events JSON array");
    CHECK(contains(json, "\"event_id\":42"), "missing raw event id");
    CHECK(contains(json, "\"host_submit_begin_us\":100"), "missing raw host submit begin");
    CHECK(contains(json, "\"host_submit_end_us\":110"), "missing raw host submit end");
    CHECK(contains(json, "\"device_submit_ns\":1000"), "missing raw device submit");
    CHECK(contains(json, "\"device_start_ns\":1200"), "missing raw device start");
    CHECK(contains(json, "\"device_end_ns\":1800"), "missing raw device end");
    CHECK(!contains(json, "command_submit_ns"), "raw JSON should not use legacy command_submit_ns field");
    CHECK(!contains(json, "command_start_ns"), "raw JSON should not use legacy command_start_ns field");
    CHECK(!contains(json, "command_end_ns"), "raw JSON should not use legacy command_end_ns field");
    CHECK(contains(json, "\"duration_ns\":600"), "missing raw duration");
    CHECK(contains(json, "\"timestamp_status\":\"test\""), "missing raw timestamp status");
    CHECK(contains(json, "\"file\":\"tests/test-sycl-kernel-profiler.cpp\""), "missing raw file");
    CHECK(contains(json, "\"function\":\"main\""), "missing raw function");
    CHECK(contains(json, "\"graph_recorded\":1"), "missing raw graph flag");
    CHECK(!contains(json, "\"node_op\""), "node context should be absent before a node scope is active");

    {
        ggml_sycl_kernel_profile_node_scope node_scope(5, 42, 1374, "MUL_MAT_ID", "ffn_gate_exps");
        ggml_sycl_kernel_profile_add_raw_event_for_test(slow, 43, 200, 210, 2000, 2200, 2600, "test-node",
                                                        "tests/test-sycl-kernel-profiler.cpp", 124, "main", false);
    }
    const std::string node_json = ggml_sycl_kernel_profile_format_json_for_test();
    CHECK(contains(node_json, "\"node_step\":5"), "missing raw node step");
    CHECK(contains(node_json, "\"node_idx\":42"), "missing raw node index");
    CHECK(contains(node_json, "\"node_count\":1374"), "missing raw node count");
    CHECK(contains(node_json, "\"node_op\":\"MUL_MAT_ID\""), "missing raw node op");
    CHECK(contains(node_json, "\"node_tensor\":\"ffn_gate_exps\""), "missing raw node tensor");
    CHECK(!contains(ggml_sycl_kernel_profile_format_csv_for_test(), "node_op"),
          "node context should not enter CSV keys");

    char stack_op[32]     = "STACK_OP";
    char stack_tensor[32] = "stack_tensor";
    {
        ggml_sycl_kernel_profile_node_scope node_scope(6, 43, 1374, stack_op, stack_tensor);
        ggml_sycl_kernel_profile_add_raw_event_for_test(slow, 44, 300, 310, 3000, 3200, 3600, "test-node-owned",
                                                        "tests/test-sycl-kernel-profiler.cpp", 125, "main", false);
    }
    std::strcpy(stack_op, "MUTATED");
    std::strcpy(stack_tensor, "mutated_tensor");
    const std::string owned_node_json = ggml_sycl_kernel_profile_format_json_for_test();
    CHECK(contains(owned_node_json, "\"node_op\":\"STACK_OP\""), "raw node op was not snapshotted");
    CHECK(contains(owned_node_json, "\"node_tensor\":\"stack_tensor\""), "raw node tensor was not snapshotted");
    CHECK(!contains(owned_node_json, "\"node_op\":\"MUTATED\""), "raw node op kept a mutable pointer");
    CHECK(!contains(owned_node_json, "\"node_tensor\":\"mutated_tensor\""),
          "raw node tensor kept a mutable pointer");

    cfg.raw_events = false;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);
    const std::string json_without_raw_events = ggml_sycl_kernel_profile_format_json_for_test();
    CHECK(contains(json_without_raw_events, "\"kernels\""), "raw-events-disabled JSON lost kernels array");
    CHECK(!contains(json_without_raw_events, "\"raw_events\""), "raw_events=false should omit raw events JSON array");
    cfg.raw_events = true;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);

    const std::string summary = ggml_sycl_kernel_profile_format_summary_for_test(2);
    CHECK(summary.find("mxfp4.gateup.packed_q8_m2") < summary.find("sycl.memcpy.graph_safe"),
          "summary not sorted by total time");

    cfg.flush_mode = ggml_sycl_kernel_profile_flush_mode::NONE;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);
    CHECK(!ggml_sycl_kernel_profile_effective_wait_for_test(true), "flush=none should not force waiting");

    cfg.flush_mode = ggml_sycl_kernel_profile_flush_mode::WINDOW;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);
    CHECK(ggml_sycl_kernel_profile_effective_wait_for_test(false), "flush=window should force waiting");

    cfg.flush_mode = ggml_sycl_kernel_profile_flush_mode::FINAL;
    ggml_sycl_kernel_profile_set_config_for_test(cfg);
    CHECK(ggml_sycl_kernel_profile_effective_wait_for_test(true), "flush=final should preserve requested wait=true");
    CHECK(!ggml_sycl_kernel_profile_effective_wait_for_test(false), "flush=final should preserve requested wait=false");

    ggml_sycl_kernel_profile_reset_for_test();
    int                     submit_calls = 0;
    ggml_sycl_profile_label wrapper_label{};
    wrapper_label.name       = "unit.wrapper.submit";
    wrapper_label.category   = "unit";
    wrapper_label.queue_kind = "compute";
    wrapper_label.metadata   = "case=disabled";

    ggml_sycl_kernel_profile_config disabled_cfg{};
    disabled_cfg.enabled = false;
    ggml_sycl_kernel_profile_set_config_for_test(disabled_cfg);
    const int disabled_result = ggml_sycl_profile_submit_for_test(wrapper_label, [&]() {
        submit_calls++;
        return 7;
    });
    CHECK(disabled_result == 7, "disabled wrapper did not return lambda result");
    CHECK(submit_calls == 1, "disabled wrapper did not call lambda exactly once");
    CHECK(ggml_sycl_kernel_profile_format_csv_for_test().find("unit.wrapper.submit") == std::string::npos,
          "disabled wrapper recorded a profile row");

    ggml_sycl_kernel_profile_config enabled_cfg{};
    enabled_cfg.enabled = true;
    ggml_sycl_kernel_profile_set_config_for_test(enabled_cfg);
    wrapper_label.metadata   = "case=enabled";
    const int enabled_result = ggml_sycl_profile_submit_for_test(wrapper_label, [&]() {
        submit_calls++;
        return 11;
    });
    CHECK(enabled_result == 11, "enabled wrapper did not return lambda result");
    CHECK(submit_calls == 2, "enabled wrapper did not call lambda exactly once");
    CHECK(contains(ggml_sycl_kernel_profile_format_csv_for_test(), "unit.wrapper.submit,unit,case=enabled"),
          "enabled wrapper did not record test row");

    ggml_sycl_kernel_profile_reset_for_test();
    return 0;
}
