#pragma once

#include "ggml.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ggml_sycl {

enum class e2e_tg_stage : uint8_t {
    DISPATCH = 0,
    CPU_DISPATCH,
    NON_MOE_MATMUL,
    MOE,
    ATTENTION,
    KV,
    ELEMENTWISE,
    GRAPH,
    CACHE,
    TRANSFER,
    OTHER,
    COUNT,
};

struct e2e_tg_stage_accum {
    uint64_t    calls     = 0;
    double      host_us   = 0.0;
    double      device_us = 0.0;
    uint64_t    bytes     = 0;
    std::string last_path = "unknown";
};

struct e2e_tg_profile_snapshot {
    uint64_t                                                                 tokens    = 0;
    uint64_t                                                                 ops       = 0;
    uint64_t                                                                 moe_calls = 0;
    std::array<e2e_tg_stage_accum, static_cast<size_t>(e2e_tg_stage::COUNT)> stages{};
};

bool         e2e_tg_profile_enabled_from_env(const char * env);
bool         e2e_tg_profile_enabled();
const char * e2e_tg_stage_name(e2e_tg_stage stage);
e2e_tg_stage e2e_tg_stage_from_op(ggml_op op, const char * tensor_name);
void         e2e_tg_profile_record(e2e_tg_stage stage,
                                   const char * path,
                                   double       host_us,
                                   double       device_us = 0.0,
                                   uint64_t     bytes     = 0,
                                   uint64_t     calls     = 1);
void         e2e_tg_profile_record_cache_event(const char * path, uint64_t bytes, double host_us);
void         e2e_tg_profile_record_transfer(const char * path, uint64_t bytes, double host_us, double device_us);
void         e2e_tg_profile_flush_if_ready(FILE * out = stderr);
void         e2e_tg_profile_force_flush(FILE * out = stderr);
void         e2e_tg_profile_reset_for_tests();
e2e_tg_profile_snapshot e2e_tg_profile_snapshot_for_tests();
void                    e2e_tg_profile_flush_for_tests(FILE * out);

class e2e_tg_scope {
  public:
    e2e_tg_scope(e2e_tg_stage stage, const char * path, bool enabled = e2e_tg_profile_enabled()) :
        enabled_(enabled),
        stage_(stage) {
        if (enabled_) {
            path_  = path && path[0] != '\0' ? path : "unknown";
            start_ = clock::now();
        }
    }

    ~e2e_tg_scope() {
        if (!enabled_) {
            return;
        }
        const auto   end     = clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(end - start_).count();
        e2e_tg_profile_record(stage_, path_.c_str(), host_us, 0.0, 0, 1);
    }

    e2e_tg_scope(const e2e_tg_scope &)             = delete;
    e2e_tg_scope & operator=(const e2e_tg_scope &) = delete;

  private:
    using clock = std::chrono::high_resolution_clock;

    bool              enabled_ = false;
    e2e_tg_stage      stage_   = e2e_tg_stage::OTHER;
    std::string       path_;
    clock::time_point start_{};
};

}  // namespace ggml_sycl
