#pragma once

// The MoE executor's cost (llama.cpp-36j0, owner ruling 2026-10-19): for a MUL_MAT_ID whose expert entries are
// host-resident, one layer of one micro-batch is run on the CPU (CpuExpertPool) or its host experts are staged
// through the VRAM ring and run on the device. Plain values, no SYCL: test-moe-exec-cost pins it with no device.

#include "ggml.h"
#include "kv-room-cost.hpp"  // the per-type CPU expert rate table moe_exec_rates_seed folds

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace ggml_sycl {

enum class moe_exec_policy : uint8_t { AUTO, CPU, RING };
enum class moe_exec_choice : uint8_t { NOT_CONSULTED, CPU, RING };
enum class moe_exec_regime : uint8_t { NONE, STREAM_ALL, ROUTED_ONLY };
enum class moe_exec_reason : uint8_t { NONE, FORCED, CHEAPER, RING_UNAVAILABLE, DECODE, NO_HOST_EXPERTS };

// The measured rates the decision divides by. Zero means "not measured": the caller substitutes the seed.
struct moe_exec_rates {
    double h2d_gbps         = 0.0;  // pinned host -> device, this device (Task 3 probe)
    double cpu_gbps         = 0.0;  // CPU expert kernel, the layer's byte-weighted mix (kv-room-cost table seed)
    double reorder_gbps     = 0.0;  // AOS -> SOA on the device
    double dev_gflops       = 0.0;  // device expert GEMM rate for the layer's types
    double cpu_gflops       = 0.0;  // CPU expert GEMM rate (compute bound at large T)
    double cpu_fixed_us     = 0.0;  // per-layer CPU orchestration cost (submit, join, scatter)
    double dev_fixed_us     = 0.0;  // per-layer device cost (ptr table, launches)
    double stream_all_p_min = 0.9;  // p_touch at or above which every host expert streams in fixed order
};

struct moe_exec_layer_inputs {
    uint32_t n_tokens          = 0;  // rows of this micro-batch reaching the layer
    int      n_expert          = 0;
    int      n_expert_used     = 0;
    size_t   host_expert_bytes = 0;    // this layer's host-resident triplet bytes
    int      n_host_experts    = 0;
    int64_t  K                 = 0;    // n_embd for gate/up, n_ff for down (use the larger: an upper bound)
    int64_t  N                 = 0;    // K and N are n_embd and n_ff in either order: only the product is used
    double   window_us         = 0.0;  // the previous layer's compute window the copies can hide under
    size_t   ring_bytes_free   = 0;    // ring bytes the layer may use (0 = no ring)
    size_t   triplet_bytes_max = 0;    // the largest host triplet of the layer
};

struct moe_exec_decision {
    moe_exec_choice choice    = moe_exec_choice::NOT_CONSULTED;
    moe_exec_regime regime    = moe_exec_regime::NONE;
    moe_exec_reason reason    = moe_exec_reason::NONE;
    double          p_touch   = 0.0;
    double          t_cpu_us  = 0.0;
    double          t_ring_us = 0.0;
};

// The share of a layer's experts at least one of T tokens routes to, top-k of E: 1 - (1 - k/E)^T.
static inline double moe_exec_p_touch(int n_expert_used, int n_expert, uint32_t n_tokens) {
    if (n_tokens == 0 || n_expert <= 0 || n_expert_used <= 0) {
        return 0.0;
    }
    const double miss = 1.0 - std::min(1.0, static_cast<double>(n_expert_used) / n_expert);
    return 1.0 - std::pow(miss, static_cast<double>(n_tokens));
}

// 2 * K * N flops per routed row per matrix, three matrices, k rows per token.
static inline double moe_exec_layer_gflop(const moe_exec_layer_inputs & in) {
    return 2.0 * 3.0 * static_cast<double>(in.K) * static_cast<double>(in.N) * in.n_expert_used * in.n_tokens / 1e9;
}

// The five rates the decision divides by must be measured (or seeded): a zero would turn a cost into inf or NaN.
static inline bool moe_exec_rates_valid(const moe_exec_rates & r) {
    const double v[] = { r.h2d_gbps, r.cpu_gbps, r.reorder_gbps, r.dev_gflops, r.cpu_gflops };
    for (double x : v) {
        if (!std::isfinite(x) || x <= 0.0) {
            return false;
        }
    }
    return true;
}

static inline moe_exec_decision moe_exec_weigh(const moe_exec_layer_inputs & in,
                                               const moe_exec_rates &        r,
                                               moe_exec_policy               policy) {
    moe_exec_decision d;
    if (in.host_expert_bytes == 0 || in.n_host_experts == 0) {
        d.reason = moe_exec_reason::NO_HOST_EXPERTS;
        return d;
    }
    if (in.n_tokens <= 1) {  // n_tokens <= 1: decode, or an empty batch
        d.choice = moe_exec_choice::CPU;
        d.reason = moe_exec_reason::DECODE;
        return d;
    }
    GGML_ASSERT(moe_exec_rates_valid(r));
    d.p_touch           = moe_exec_p_touch(in.n_expert_used, in.n_expert, in.n_tokens);
    d.regime            = d.p_touch >= r.stream_all_p_min ? moe_exec_regime::STREAM_ALL : moe_exec_regime::ROUTED_ONLY;
    // Bytes moved: every host expert under STREAM_ALL, the touched share otherwise (GB/s is 1e3 bytes per us).
    const double bytes  = d.regime == moe_exec_regime::STREAM_ALL ? static_cast<double>(in.host_expert_bytes) :
                                                                    d.p_touch * in.host_expert_bytes;
    const double gflop  = moe_exec_layer_gflop(in) * (static_cast<double>(in.n_host_experts) / in.n_expert);
    const double cpu_bw = d.p_touch * in.host_expert_bytes / (r.cpu_gbps * 1e3);
    const double cpu_fl = gflop / r.cpu_gflops * 1e6;
    d.t_cpu_us          = std::max(cpu_bw, cpu_fl) + r.cpu_fixed_us;
    // Only STREAM_ALL knows its experts before the layer runs, so only it can copy layer-ahead under the previous
    // layer's window; a ROUTED_ONLY layer's ids exist after its own router, so its copy is exposed.
    const double copy_raw = bytes / (r.h2d_gbps * 1e3);
    const double copy     = d.regime == moe_exec_regime::STREAM_ALL ? std::max(0.0, copy_raw - in.window_us) : copy_raw;
    const double reord    = bytes / (r.reorder_gbps * 1e3);
    const double dev_fl   = gflop / r.dev_gflops * 1e6;
    d.t_ring_us           = copy + reord + dev_fl + r.dev_fixed_us;
    if (policy == moe_exec_policy::CPU) {
        d.choice = moe_exec_choice::CPU;
        d.reason = moe_exec_reason::FORCED;
        return d;
    }
    if (in.ring_bytes_free < in.triplet_bytes_max || in.ring_bytes_free == 0) {
        d.choice = moe_exec_choice::CPU;
        d.reason = moe_exec_reason::RING_UNAVAILABLE;
        return d;
    }
    if (policy == moe_exec_policy::RING) {
        d.choice = moe_exec_choice::RING;
        d.reason = moe_exec_reason::FORCED;
        return d;
    }
    d.choice = d.t_ring_us < d.t_cpu_us ? moe_exec_choice::RING : moe_exec_choice::CPU;
    d.reason = moe_exec_reason::CHEAPER;
    return d;
}

typedef const char * (*moe_exec_getenv_fn)(const char * name);

// The w013 rule (ggml_sycl_plan_cost_param): unset, unparsable, non-finite or <= 0 keeps the default.
static inline double moe_exec_env_param(moe_exec_getenv_fn getenv_fn, const char * name, double def) {
    const char * env = getenv_fn ? getenv_fn(name) : nullptr;
    if (env == nullptr) {
        return def;
    }
    char *       end   = nullptr;
    const double value = std::strtod(env, &end);
    if (end == env || !std::isfinite(value) || value <= 0.0) {
        return def;
    }
    return value;
}

// The seed record for a layer whose host-resident experts are `host_mix` (type, bytes). cpu_gbps is the byte-weighted
// harmonic mean over the measured types, as kv_room_expert_types::add folds them; the room's default when none is.
static inline moe_exec_rates moe_exec_rates_seed(const std::vector<std::pair<ggml_type, size_t>> & host_mix) {
    moe_exec_rates       r;
    kv_room_expert_types types;
    for (const auto & tb : host_mix) {
        types.add(tb.first, tb.second);
    }
    const size_t measured = types.bytes - types.unmeasured_bytes;
    r.cpu_gbps = measured > 0 && types.measured_us > 0.0 ? static_cast<double>(measured) / (types.measured_us * 1e3) :
                                                           PLACEMENT_CPU_EXPERT_GBPS_DEFAULT;
    r.h2d_gbps = 20.0;         // seed; the Task 3 probe replaces it (measured 26.5 B70 / 27.1 B50, 8bgi c-nn1q)
    r.reorder_gbps = 200.0;    // seed; the on-device AOS->SOA reorder (B70 D2D 870 GB/s, kernel-bound well below)
    r.dev_gflops   = 20000.0;  // seed; Task 9 seeds per type from the batched prompt executor measurement
    r.cpu_gflops   = 1000.0;   // seed; 22 threads of AVX2 vec_dot (cego)
    r.cpu_fixed_us = 500.0;    // seed; per-layer submit/join/scatter (8bgi orchestration map)
    r.dev_fixed_us = 800.0;    // seed; ptr-table upload + per-expert launches
    return r;
}

// Env overrides for the four rates that have one. getenv_fn is injected so a test needs no real environment.
static inline void moe_exec_rates_apply_env(moe_exec_rates & r, moe_exec_getenv_fn getenv_fn) {
    r.h2d_gbps     = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_H2D_GBPS", r.h2d_gbps);
    r.cpu_gbps     = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_CPU_GBPS", r.cpu_gbps);
    r.dev_gflops   = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_DEV_GFLOPS", r.dev_gflops);
    r.reorder_gbps = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_REORDER_GBPS", r.reorder_gbps);
}

}  // namespace ggml_sycl
