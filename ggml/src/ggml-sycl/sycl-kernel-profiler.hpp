#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>

enum class ggml_sycl_kernel_profile_output_format : uint8_t {
    STDERR = 0,
    CSV    = 1,
    JSON   = 2,
    BOTH   = 3,
};

enum class ggml_sycl_kernel_profile_flush_mode : uint8_t {
    FINAL  = 0,
    WINDOW = 1,
    NONE   = 2,
};

struct ggml_sycl_kernel_profile_config {
    bool                                   enabled       = false;
    ggml_sycl_kernel_profile_output_format output_format = ggml_sycl_kernel_profile_output_format::CSV;
    ggml_sycl_kernel_profile_flush_mode    flush_mode    = ggml_sycl_kernel_profile_flush_mode::FINAL;
    int                                    top_n         = 40;
    bool                                   raw_events    = false;
    std::string                            output_path;
};

struct ggml_sycl_profile_label {
    const char * name       = "unknown";
    const char * category   = "unknown";
    const char * queue_kind = "unknown";
    const char * metadata   = "";
    int          device     = -1;
    size_t       bytes      = 0;
};

bool                            ggml_sycl_kernel_profile_enabled();
ggml_sycl_kernel_profile_config ggml_sycl_kernel_profile_config_from_env();
void ggml_sycl_kernel_profile_record_event(const ggml_sycl_profile_label & label, const sycl::event & event);
void ggml_sycl_kernel_profile_flush(bool wait_for_events, const char * reason);

void ggml_sycl_kernel_profile_reset_for_test();
void ggml_sycl_kernel_profile_set_config_for_test(const ggml_sycl_kernel_profile_config & cfg);
void ggml_sycl_kernel_profile_add_sample_for_test(const ggml_sycl_profile_label & label, uint64_t duration_ns);
void ggml_sycl_kernel_profile_add_failed_timestamp_for_test(const ggml_sycl_profile_label & label, bool graph_recorded);
std::string ggml_sycl_kernel_profile_format_csv_for_test();
std::string ggml_sycl_kernel_profile_format_json_for_test();
std::string ggml_sycl_kernel_profile_format_summary_for_test(int top_n);
bool        ggml_sycl_kernel_profile_effective_wait_for_test(bool requested_wait);
