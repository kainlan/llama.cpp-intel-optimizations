#include "e2e-profile.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ggml_sycl {
namespace {

thread_local e2e_tg_profile_snapshot g_e2e_tg_profile;

bool tensor_name_contains(const char * name, const char * needle) {
    return name && needle && std::strstr(name, needle) != nullptr;
}

bool tensor_name_is_kv(const char * name) {
    return tensor_name_contains(name, "cache_k") || tensor_name_contains(name, "cache_v") ||
           tensor_name_contains(name, "kv") || tensor_name_contains(name, "KQ_mask");
}

}  // namespace

bool e2e_tg_profile_enabled_from_env(const char * env) {
    return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
}

bool e2e_tg_profile_enabled() {
    static const bool enabled = e2e_tg_profile_enabled_from_env(std::getenv("GGML_SYCL_E2E_TG_PROFILE"));
    return enabled;
}

const char * e2e_tg_stage_name(e2e_tg_stage stage) {
    switch (stage) {
        case e2e_tg_stage::DISPATCH:
            return "dispatch";
        case e2e_tg_stage::CPU_DISPATCH:
            return "cpu_dispatch";
        case e2e_tg_stage::NON_MOE_MATMUL:
            return "non_moe_matmul";
        case e2e_tg_stage::MOE:
            return "moe";
        case e2e_tg_stage::ATTENTION:
            return "attention";
        case e2e_tg_stage::KV:
            return "kv";
        case e2e_tg_stage::ELEMENTWISE:
            return "elementwise";
        case e2e_tg_stage::GRAPH:
            return "graph";
        case e2e_tg_stage::CACHE:
            return "cache";
        case e2e_tg_stage::TRANSFER:
            return "transfer";
        case e2e_tg_stage::OTHER:
            return "other";
        case e2e_tg_stage::COUNT:
            return "count";
    }
    return "unknown";
}

e2e_tg_stage e2e_tg_stage_from_op(ggml_op op, const char * tensor_name) {
    if (tensor_name_is_kv(tensor_name)) {
        return e2e_tg_stage::KV;
    }
    switch (op) {
        case GGML_OP_MUL_MAT_ID:
            return e2e_tg_stage::MOE;
        case GGML_OP_FLASH_ATTN_EXT:
            return e2e_tg_stage::ATTENTION;
        case GGML_OP_MUL_MAT:
            return e2e_tg_stage::NON_MOE_MATMUL;
        case GGML_OP_SET_ROWS:
        case GGML_OP_SET_ROWS_PAGED:
            return e2e_tg_stage::KV;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_GLU:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return e2e_tg_stage::ELEMENTWISE;
        default:
            return e2e_tg_stage::OTHER;
    }
}

void e2e_tg_profile_record(e2e_tg_stage stage,
                           const char * path,
                           double       host_us,
                           double       device_us,
                           uint64_t     bytes,
                           uint64_t     calls) {
    if (stage == e2e_tg_stage::COUNT) {
        stage = e2e_tg_stage::OTHER;
    }
    auto & slot = g_e2e_tg_profile.stages[static_cast<size_t>(stage)];
    slot.calls += calls;
    slot.host_us += std::max(0.0, host_us);
    slot.device_us += std::max(0.0, device_us);
    slot.bytes += bytes;
    slot.last_path = path && path[0] != '\0' ? path : "unknown";
    g_e2e_tg_profile.ops += calls;
    if (stage == e2e_tg_stage::MOE) {
        g_e2e_tg_profile.moe_calls += calls;
    }
}

void e2e_tg_profile_record_cache_event(const char * path, uint64_t bytes, double host_us) {
    e2e_tg_profile_record(e2e_tg_stage::CACHE, path, host_us, 0.0, bytes, 1);
}

void e2e_tg_profile_record_transfer(const char * path, uint64_t bytes, double host_us, double device_us) {
    e2e_tg_profile_record(e2e_tg_stage::TRANSFER, path, host_us, device_us, bytes, 1);
}

void e2e_tg_profile_force_flush(FILE * out) {
    if (!out) {
        out = stderr;
    }
    double total_host_us   = 0.0;
    double total_device_us = 0.0;
    for (const auto & stage : g_e2e_tg_profile.stages) {
        total_host_us += stage.host_us;
        total_device_us += stage.device_us;
    }
    if (g_e2e_tg_profile.ops == 0) {
        return;
    }
    g_e2e_tg_profile.tokens += 1;
    std::fprintf(out,
                 "[SYCL-E2E-TG-PROFILE] tokens=%llu ops=%llu moe_calls=%llu total_host=%.3f ms total_device=%.3f ms\n",
                 (unsigned long long) g_e2e_tg_profile.tokens, (unsigned long long) g_e2e_tg_profile.ops,
                 (unsigned long long) g_e2e_tg_profile.moe_calls, total_host_us / 1000.0, total_device_us / 1000.0);
    for (size_t i = 0; i < static_cast<size_t>(e2e_tg_stage::COUNT); ++i) {
        const auto & stage_accum = g_e2e_tg_profile.stages[i];
        if (stage_accum.calls == 0) {
            continue;
        }
        std::fprintf(out,
                     "[SYCL-E2E-TG-STAGE] stage=%s calls=%llu host=%.3f ms device=%.3f ms bytes=%llu last_path=%s\n",
                     e2e_tg_stage_name(static_cast<e2e_tg_stage>(i)), (unsigned long long) stage_accum.calls,
                     stage_accum.host_us / 1000.0, stage_accum.device_us / 1000.0,
                     (unsigned long long) stage_accum.bytes, stage_accum.last_path.c_str());
    }
    g_e2e_tg_profile = {};
}

void e2e_tg_profile_flush_if_ready(FILE * out) {
    if (g_e2e_tg_profile.moe_calls >= 72) {
        e2e_tg_profile_force_flush(out);
    }
}

void e2e_tg_profile_reset_for_tests() {
    g_e2e_tg_profile = {};
}

e2e_tg_profile_snapshot e2e_tg_profile_snapshot_for_tests() {
    return g_e2e_tg_profile;
}

void e2e_tg_profile_flush_for_tests(FILE * out) {
    e2e_tg_profile_force_flush(out);
}

}  // namespace ggml_sycl
