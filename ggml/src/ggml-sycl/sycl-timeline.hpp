#pragma once

#include <chrono>
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

struct sycl_timeline_callsite {
    const char * file     = "unknown";
    int          line     = 0;
    const char * function = "unknown";
};

class sycl_timeline_scope {
  public:
    sycl_timeline_scope(const char *           category,
                        const char *           name,
                        const char *           metadata,
                        sycl_timeline_callsite callsite);
    ~sycl_timeline_scope();

    sycl_timeline_scope(const sycl_timeline_scope &)             = delete;
    sycl_timeline_scope & operator=(const sycl_timeline_scope &) = delete;

  private:
    bool                                  active_ = false;
    std::string                           category_;
    std::string                           name_;
    std::string                           metadata_;
    sycl_timeline_callsite                callsite_ = {};
    std::chrono::steady_clock::time_point start_time_;
};

bool                 sycl_timeline_enabled_from_env(const char * value);
bool                 sycl_timeline_enabled();
sycl_timeline_config sycl_timeline_config_from_env();
sycl_timeline_config sycl_timeline_config_from_values(const char * mode,
                                                      const char * output,
                                                      const char * token_start,
                                                      const char * token_count,
                                                      const char * max_events);
void                 sycl_timeline_record_span(const char *                          category,
                                               const char *                          name,
                                               const char *                          metadata,
                                               sycl_timeline_callsite                callsite,
                                               std::chrono::steady_clock::time_point start_time,
                                               std::chrono::steady_clock::time_point end_time);
void                 sycl_timeline_flush(const char * reason);
std::string          sycl_timeline_format_json_for_tests();
void                 sycl_timeline_reset_for_tests();
void                 sycl_timeline_set_config_for_tests(const sycl_timeline_config & cfg);

}  // namespace ggml_sycl

#define GGML_SYCL_TL_CONCAT_(a, b)       a##b
#define GGML_SYCL_TL_CONCAT(a, b)        GGML_SYCL_TL_CONCAT_(a, b)
#define GGML_SYCL_TL_SCOPE_NAME(counter) GGML_SYCL_TL_CONCAT(ggml_sycl_timeline_scope_, counter)
#define GGML_SYCL_TIMELINE_SCOPE(cat, name, metadata)                    \
    ggml_sycl::sycl_timeline_scope GGML_SYCL_TL_SCOPE_NAME(__COUNTER__)( \
        (cat), (name), (metadata), ggml_sycl::sycl_timeline_callsite{ __FILE__, __LINE__, __func__ })
