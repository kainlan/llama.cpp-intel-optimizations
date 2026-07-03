#pragma once

#include <cstdint>
#include <string>

namespace ggml_sycl {

enum class sycl_timeline_mode : uint8_t {
    OFF = 0,
    SUMMARY,
    TIMELINE,
    TIMELINE_EVENTS,
};

struct sycl_timeline_config {
    bool               enabled = false;
    sycl_timeline_mode mode    = sycl_timeline_mode::OFF;
    std::string        output_path;
    int                token_start = 0;
    int                token_count = 0;
    int                max_events  = 200000;
};

bool                 sycl_timeline_enabled_from_env(const char * value);
bool                 sycl_timeline_enabled();
sycl_timeline_config sycl_timeline_config_from_env();
sycl_timeline_config sycl_timeline_config_from_values(const char * mode,
                                                      const char * output,
                                                      const char * token_start,
                                                      const char * token_count,
                                                      const char * max_events);
void                 sycl_timeline_reset_for_tests();
void                 sycl_timeline_set_config_for_tests(const sycl_timeline_config & cfg);

}  // namespace ggml_sycl
