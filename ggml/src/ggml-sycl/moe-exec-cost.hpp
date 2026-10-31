#pragma once

// The MoE executor's cost (llama.cpp-36j0, owner ruling 2026-10-19): for a MUL_MAT_ID whose expert entries are
// host-resident, one layer of one micro-batch is run on the CPU (CpuExpertPool) or its host experts are staged
// through the VRAM ring and run on the device. Plain values, no SYCL: test-moe-exec-cost pins it with no device.

#include "ggml.h"
#include "kv-room-cost.hpp"     // the per-type CPU expert rate table moe_exec_rates_seed folds
#include "moe-mmvq-tables.hpp"  // moe_pp_batched_takes_staged_type: the types the executor takes staged

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace ggml_sycl {

enum class moe_exec_policy : uint8_t { AUTO, CPU, RING };
enum class moe_exec_choice : uint8_t { NOT_CONSULTED, CPU, RING };
enum class moe_exec_regime : uint8_t { NONE, STREAM_ALL, ROUTED_ONLY };
enum class moe_exec_reason : uint8_t { NONE, FORCED, CHEAPER, RING_UNAVAILABLE, DECODE, NO_HOST_EXPERTS, DECLINED };

// The measured rates the decision divides by. Zero means "not measured": the caller substitutes the seed.
struct moe_exec_rates {
    double h2d_gbps         = 0.0;  // pinned host -> device, this device (Task 3 probe)
    double cpu_gbps         = 0.0;  // CPU expert kernel, the layer's byte-weighted mix (kv-room-cost table seed)
    double dev_gflops       = 0.0;  // device expert GEMM rate for the layer's types
    double cpu_gflops       = 0.0;  // CPU expert GEMM rate (compute bound at large T)
    double cpu_fixed_us     = 0.0;  // per-layer CPU orchestration cost (submit, join, scatter)
    double dev_fixed_us     = 0.0;  // per-layer device cost (ptr table, launches)
    double stream_all_p_min = 0.9;  // p_touch at or above which every host expert streams in fixed order
};

struct moe_exec_layer_inputs {
    uint32_t n_tokens             = 0;  // rows of this micro-batch reaching the layer
    int      n_expert             = 0;
    int      n_expert_used        = 0;
    size_t   host_expert_bytes    = 0;    // this layer's host-resident triplet bytes
    int      n_host_experts       = 0;
    int64_t  K                    = 0;    // n_embd for gate/up, n_ff for down (use the larger: an upper bound)
    int64_t  N                    = 0;    // K and N are n_embd and n_ff in either order: only the product is used
    size_t   ring_bytes_free      = 0;    // ring bytes the layer may use (0 = no ring)
    size_t   stage_unit_bytes_max = 0;    // the largest one matrix the ring stages for the layer (one slot)
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

// The four rates the decision divides by must be measured (or seeded): a zero would turn a cost into inf or NaN.
static inline bool moe_exec_rates_valid(const moe_exec_rates & r) {
    const double v[] = { r.h2d_gbps, r.cpu_gbps, r.dev_gflops, r.cpu_gflops };
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
    // The ring copies every byte it moves and the executor reads each slot in place (AOS), so nothing is reordered.
    // The layer is weighed at its own first op and no copy hides under an earlier layer (there is no cross-layer
    // prefetch), so the whole copy is priced.
    const double copy     = bytes / (r.h2d_gbps * 1e3);
    const double dev_fl   = gflop / r.dev_gflops * 1e6;
    d.t_ring_us           = copy + dev_fl + r.dev_fixed_us;
    if (policy == moe_exec_policy::CPU) {
        d.choice = moe_exec_choice::CPU;
        d.reason = moe_exec_reason::FORCED;
        return d;
    }
    if (in.ring_bytes_free < in.stage_unit_bytes_max || in.ring_bytes_free == 0) {
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

// The device expert executor's seed rate, GFLOP/s, for one ggml type: the batched prompt executor as a whole (GEMM,
// staging and dequant or repack), since that whole is what a staged layer costs on the device. The persisted per-device
// record replaces a seed once measured.
//   MXFP4: Task 9 (llama.cpp-g4uy, 2026-10-21), B50, GPT-OSS 20B through bench-guard with GGML_SYCL_KERNEL_PROFILE=1:
//          the batched XMX executor as a whole (gemm.execute + repack.tiled + repack.soa + stage: 12.25 s over
//          48.9 TFLOP); the GEMM kernel alone is 10070.
//   The f16 arm's types: measured by the lead on d52110c80 (llama.cpp-348c, 2026-10-23), B70, llama-bench -v with
//   GGML_SYCL_KERNEL_PROFILE=1 at auto -ub 512, re-seeded in llama.cpp-3460.
//   Q4_0:  Qwen1.5-MoE Q4_0, 20 ubatches of 512 = 17.0 TFLOP: moe.pp.gemm.execute 1.383 s, stage.act 0.226 s,
//          stage.scatter 0.266 s, dequant.q4_0 0.009 s, so about 12,300 GEMM-only and about 9,000 as a whole.
//   other: every other type the executor takes from a staged slot (moe_pp_batched_takes_staged_type, the one type
//          question the staged claim asks too). Qwen3.8 GSQ-RCO runs one f16 GEMM for iq2_xxs, iq2_xs, iq2_s, iq3_s,
//          iq4_nl and q2_0, with the per-type dequant under 1% of the time: gemm 0.248 s, stage 0.163 s, dequant
//          0.001 s, so about 8,200 GEMM-only and about 5,000 as a whole.
//   A type the executor does not take from a staged slot keeps the per-row MMVQ rate (Task 9: the Q4_0 down matrix
//   measured 590). IQ1_S, IQ1_M and IQ4_XS are among them: the f16 arm dequantizes them, but they have no MMVQ-ID
//   kernel, so the staged claim refuses them and the ring does not stage them (llama.cpp-w4qk).
static constexpr double MOE_EXEC_DEV_GFLOPS_PER_ROW_SEED = 600.0;
static constexpr double MOE_EXEC_DEV_GFLOPS_F16_ARM_SEED = 5000.0;

static inline double moe_exec_dev_gflops_seed(ggml_type t) {
    if (t == GGML_TYPE_MXFP4) {
        return 3990.0;
    }
    if (!moe_pp_batched_takes_staged_type(t)) {
        return MOE_EXEC_DEV_GFLOPS_PER_ROW_SEED;
    }
    return t == GGML_TYPE_Q4_0 ? 9000.0 : MOE_EXEC_DEV_GFLOPS_F16_ARM_SEED;
}

// The seed record for a layer whose host-resident experts are `host_mix` (type, bytes). cpu_gbps is the byte-weighted
// harmonic mean over the measured types, as kv_room_expert_types::add folds them; the room's default when none is.
// dev_gflops is the byte-weighted harmonic mean of moe_exec_dev_gflops_seed (bytes / sum(bytes_i / rate_i): the cost
// is bytes-proportional time), the per-row 600 for an empty mix.
static inline moe_exec_rates moe_exec_rates_seed(const std::vector<std::pair<ggml_type, size_t>> & host_mix) {
    kv_room_expert_types types;
    for (const auto & tb : host_mix) {
        types.add(tb.first, tb.second);
    }
    const size_t measured  = types.bytes - types.unmeasured_bytes;
    const double cpu_gbps  = measured > 0 && types.measured_us > 0.0 ?
                                 static_cast<double>(measured) / (types.measured_us * 1e3) :
                                 PLACEMENT_CPU_EXPERT_GBPS_DEFAULT;
    double       dev_bytes = 0.0;
    double       dev_us    = 0.0;  // bytes_i / rate_i summed, in units of 1 / (GFLOP/s)
    for (const auto & tb : host_mix) {
        dev_bytes += static_cast<double>(tb.second);
        dev_us += static_cast<double>(tb.second) / moe_exec_dev_gflops_seed(tb.first);
    }
    const double dev_gflops = dev_bytes > 0.0 ? dev_bytes / dev_us : MOE_EXEC_DEV_GFLOPS_PER_ROW_SEED;

    moe_exec_rates r;
    r.cpu_gbps     = cpu_gbps;
    r.h2d_gbps     = 20.0;        // seed; the Task 3 probe replaces it (measured 26.5 B70 / 27.1 B50, 8bgi c-nn1q)
    r.dev_gflops   = dev_gflops;  // seed; per type, moe_exec_dev_gflops_seed
    r.cpu_gflops   = 1000.0;      // seed; 22 threads of AVX2 vec_dot (cego)
    r.cpu_fixed_us = 500.0;       // seed; per-layer submit/join/scatter (8bgi orchestration map)
    r.dev_fixed_us = 800.0;       // seed; ptr-table upload + per-expert launches
    return r;
}

// llama.cpp-xr4i (c-vbf4, the bimodal readings): the H2D probe times each of its iterations and persists the best
// iteration after the first, not the mean. Ten probed readings of one driver key came out 25.8, 23.4, 9.3, 9.8, 9.6,
// 25.3, 24.0, 26.0, 26.2, 25.2 GB/s on two cards that idle at PCIe 2.5 GT/s x1; a mean over one block let a link that
// had not ramped inside the first window freeze 9.x into the tuning cache as the device's rate. The cost model needs
// the device's sustained copy capacity, which one slow window does not change, so the rate is the max of iterations
// 2..n (iteration 1 is a second warm-up) and the spread is returned for the log.
//
// A max still under this fraction of the seed is not a rate. The seed is 20 GB/s, set under the 26.5-27.1 GB/s that
// llama.cpp-8bgi c-nn1q measured, so half of it (10 GB/s) sits well below a healthy link and just above the 9.x GB/s
// bad readings this threshold exists to refuse. Persisting one would bias every ring-vs-cpu decision for the life of
// the cache entry, so the probe retries at the next load instead.
static constexpr double MOE_EXEC_H2D_MIN_SEED_FRACTION = 0.5;

// gbps[0..n) are the per-iteration rates in GB/s, in order. Returns the rate, or 0 when there is none: fewer than two
// iterations, no positive finite one among 2..n, or a max under MOE_EXEC_H2D_MIN_SEED_FRACTION of the seed. lo and hi
// (each optional) receive the min and max over the positive finite iterations of 2..n, 0 when there are none, whether
// or not the rate is returned, so a rejected window still shows its spread.
static inline double moe_exec_h2d_rate_from_iterations(const double * gbps,
                                                       int            n,
                                                       double         seed,
                                                       double *       lo,
                                                       double *       hi) {
    double best = 0.0;
    double low  = 0.0;
    bool   any  = false;
    for (int i = 1; i < n; ++i) {
        const double g = gbps[i];
        if (!(g > 0.0) || !std::isfinite(g)) {
            continue;
        }
        if (!any) {
            best = g;
            low  = g;
            any  = true;
        } else {
            if (g > best) {
                best = g;
            }
            if (g < low) {
                low = g;
            }
        }
    }
    if (lo != nullptr) {
        *lo = low;
    }
    if (hi != nullptr) {
        *hi = best;
    }
    if (!any || !(best >= seed * MOE_EXEC_H2D_MIN_SEED_FRACTION)) {
        return 0.0;
    }
    return best;
}

// Env overrides for the three rates that have one. getenv_fn is injected so a test needs no real environment.
static inline void moe_exec_rates_apply_env(moe_exec_rates & r, moe_exec_getenv_fn getenv_fn) {
    r.h2d_gbps   = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_H2D_GBPS", r.h2d_gbps);
    r.cpu_gbps   = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_CPU_GBPS", r.cpu_gbps);
    r.dev_gflops = moe_exec_env_param(getenv_fn, "GGML_SYCL_MOE_EXEC_DEV_GFLOPS", r.dev_gflops);
}

}  // namespace ggml_sycl
