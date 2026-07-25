//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Dispatch for SYCL Matmul
//
// This header provides a simplified, data-driven dispatch function that replaces
// the complex mul_mat dispatch logic (~200 lines) with a concise ~20 line implementation.
//
// Key features:
// - Environment variable gate (GGML_SYCL_UNIFIED_KERNEL)
// - Debug tracing (GGML_SYCL_DEBUG >= 1)
// - Integration with TuningEngine for cached/heuristic params
// - Fallback to legacy kernel for unsupported types
//

#pragma once

#include "cold-start.hpp"
#include "gpu-arch.hpp"
#include "op-context.hpp"
#include "persistent-tg-kernel.hpp"
#include "tuning-engine-impl.hpp"
#include "tuning-engine.hpp"
#include "unified-kernel.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Declared in common.hpp, defined in common.cpp (external linkage), and
// redeclared here rather than pulling common.hpp in.
//
// dispatch.hpp has exactly three includers: ggml-sycl.cpp (line 286, via the
// path-prefixed "ggml-sycl/dispatch.hpp"), tools/sycl-kernel-bench/
// benchmark_harness.hpp, and tests/test-unified-dispatch-integration.cpp —
// the last of which does not include common.hpp at all, so pulling it in here
// would end that test's independence from the backend library. The weight of
// this header matters because one of those three is the core backend TU, not
// because the list is long.
//
// dpct::dev_mgr — the other half of common.hpp's ggml_sycl_get_device() — is
// already visible transitively through unified-kernel.hpp -> unified-cache.hpp.
// common.hpp carries a pointer back to this redeclaration.
int ggml_sycl_map_device_id(int device);

namespace ggml_sycl {

// =============================================================================
// Kernel Type Enum for Dispatch Selection
// =============================================================================

/**
 * Kernel types for matmul dispatch selection.
 *
 * The dispatch system selects the optimal kernel based on:
 * - Batch size (M dimension)
 * - Weight quantization type
 * - Hardware capabilities (XMX, ESIMD dpas)
 * - Memory layout
 */
enum class KernelType {
    LEGACY,        // Legacy per-operation kernels (DMMV, MMQ, etc.)
    UNIFIED,       // Unified kernel with tuning engine
    ONEDNN,        // oneDNN FP16/BF16 path for large batches
    PERSISTENT_TG  // Persistent token generation kernel (whole-graph)
};

// =============================================================================
// Quant Type Support Check
// =============================================================================

/**
 * Check if unified kernel should be used for this quantization type.
 *
 * Unified kernel supports quantized integer types. FP16/BF16 use oneDNN.
 *
 * @param type GGML quantization type
 * @return true if unified kernel supports this type
 */
inline bool should_use_unified(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_MXFP4:
            // TODO: Add Q8_0, Q6_K, Q4_K support to unified kernel
            return true;
        default:
            return false;  // FP16, BF16, F32, Q6_K, etc. use legacy path for now
    }
}

// =============================================================================
// Environment Variable Gates
// =============================================================================

/**
 * Check if unified kernel is enabled via environment variable.
 *
 * Set GGML_SYCL_UNIFIED_KERNEL=0 to disable (default: enabled).
 *
 * @return true if unified kernel is enabled
 */
inline bool is_unified_kernel_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_UNIFIED_KERNEL");
        enabled = (!env || std::strcmp(env, "0") != 0) ? 1 : 0;  // Default: enabled
    }
    return enabled != 0;
}

/**
 * Report, once, why the unified kernel was disabled for a device.
 *
 * The B70 ran the legacy mul_mat path for an unknown period because it was
 * misclassified as Alchemist, and the message here said only THAT the unified
 * kernel was off — never which device, which family, or where that family came
 * from. Print what was actually observed so the next misdetection is
 * diagnosable from one run.
 *
 * @param device_id Device the unified kernel was disabled for
 */
inline void log_unified_kernel_disabled(int device_id) {
    // XMXConfig::from_device() returns its default (supports_esimd_dpas=false)
    // config for an out-of-range device_id without inspecting any device, so
    // this path is reachable with an id no device manager can resolve. Report
    // that instead of naming a family for a device that was never queried.
    //
    // A negative id is answered without touching dpct::dev_mgr on purpose:
    // dev_mgr::instance() enumerates every SYCL device on first use, which is
    // far too much to do just to print a count next to an id that was already
    // invalid on its face. It also keeps this branch reachable from a
    // host-only test (see tests/test-esimd-dpas-gate.cpp).
    if (device_id < 0) {
        fprintf(stderr,
                "[unified-dispatch] Unified kernel disabled: device id %d is negative, "
                "so no ESIMD dpas capability was queried\n",
                device_id);
        fprintf(stderr, "[unified-dispatch] Using legacy mul_mat kernels\n");
        fflush(stderr);
        return;
    }

    const unsigned int dpct_id = static_cast<unsigned int>(ggml_sycl_map_device_id(device_id));
    if (dpct_id >= dpct::dev_mgr::instance().device_count()) {
        fprintf(stderr,
                "[unified-dispatch] Unified kernel disabled: device %d is out of range "
                "(%u SYCL device(s) present), so no ESIMD dpas capability was queried\n",
                device_id, dpct::dev_mgr::instance().device_count());
        fprintf(stderr, "[unified-dispatch] Using legacy mul_mat kernels\n");
        fflush(stderr);
        return;
    }

    // Same resolution common.hpp's ggml_sycl_get_device() performs.
    // dpct::device_ext derives publicly from sycl::device.
    const sycl::device &  dev    = dpct::dev_mgr::instance().get_device(dpct_id);
    const sycl_gpu_family family = family_from_device(dev);

    // family_from_device() takes the architecture query first and falls back to
    // the device-name heuristic only when that query yields UNKNOWN or throws,
    // so re-running the architecture leg here reports which leg produced the
    // family printed above it.
    //
    // Standing follow-up llama.cpp-ajg9: have XMXConfig carry the family it
    // already computed, plus how it derived it. That removes this second query,
    // the device lookup above, and the ggml_sycl_map_device_id redeclaration.
    const char * source = "device-name heuristic; the architecture query threw";
    try {
        const auto arch = dev.get_info<sycl::ext::oneapi::experimental::info::device::architecture>();
        source          = family_from_architecture(arch) != sycl_gpu_family::UNKNOWN ?
                              "SYCL architecture query" :
                              "device-name heuristic; the architecture is not one this backend maps";
    } catch (const sycl::exception &) {
        // Keep the initial text — the throw is the observation.
    }

    fprintf(stderr,
            "[unified-dispatch] Unified kernel disabled: device %d (%s) family=%s (%s) "
            "lacks ESIMD dpas (ExecutionSize=16)\n",
            device_id, dev.get_info<sycl::info::device::name>().c_str(), family_name(family), source);
    fprintf(stderr,
            "[unified-dispatch] Using legacy mul_mat kernels. If this device is Xe2/Battlemage or "
            "PVC class, the family detection is wrong -- see ggml/src/ggml-sycl/gpu-arch.cpp\n");
    fflush(stderr);
}

/**
 * Check if unified kernel should be used on this device.
 *
 * The unified kernel requires ESIMD dpas support for optimal performance.
 * Without ESIMD dpas, the unified kernel falls back to a slow scalar path
 * that's much slower than the legacy mul_mat kernels.
 *
 * COMPILER LIMITATION (oneAPI 2025.3.2):
 * Even if we don't dispatch to ESIMD kernels at runtime, they get JIT compiled
 * when any kernel in the unified-kernel.cpp translation unit is submitted.
 * This causes compilation failures on platforms where dpas intrinsics aren't
 * supported (like Arc Battlemage, which is treated as XeLPG by the compiler).
 *
 * Until the compiler adds Xe2/Battlemage ESIMD dpas support, we disable the
 * unified kernel entirely on hardware that doesn't support ESIMD dpas.
 *
 * @param device_id Device to check (default: 0)
 * @return true if unified kernel should be used on this device
 */
inline bool is_unified_kernel_enabled_for_device(int device_id = 0) {
    // First check the environment variable gate
    if (!is_unified_kernel_enabled()) {
        return false;
    }

    // Check if the device supports ESIMD dpas
    // Without ESIMD dpas, the unified kernel falls back to slow paths AND
    // causes JIT compilation failures when ESIMD kernels are compiled
    ggml_sycl_unified::XMXConfig cfg = ggml_sycl_unified::XMXConfig::from_device(device_id);
    if (!cfg.supports_esimd_dpas) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            log_unified_kernel_disabled(device_id);
        }
        return false;
    }

    return true;
}

/**
 * Get debug level from environment variable.
 *
 * @return Debug level (0 = off, 1+ = enabled)
 */
inline int get_debug_level() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("GGML_SYCL_DEBUG");
        level = (env != nullptr) ? std::atoi(env) : 0;
    }
    return level;
}

/**
 * Check if persistent TG kernel is enabled via environment variable.
 *
 * Persistent token generation kernel — fuses entire TG forward pass.
 * Default: disabled. Set GGML_SYCL_PERSISTENT_TG=1 to enable.
 *
 * @return true if persistent TG kernel is enabled
 */
inline bool env_persistent_tg_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG");
        // Default OFF until the persistent policy is faster for the normal TG shape.
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

/**
 * Select the optimal kernel type for a given matmul operation.
 *
 * Decision factors:
 * - is_decode_phase: true if this is token generation (not prompt processing)
 * - batch_size: M dimension (number of output rows)
 * - n_layers: number of transformer layers
 * - hidden_dim: model hidden dimension
 * - quant_type: weight quantization type (GGML_TYPE_*)
 * - xmx_config: hardware XMX configuration
 *
 * @return Selected KernelType for dispatch
 */
inline KernelType select_kernel_type(bool                                       is_decode_phase,
                                     int64_t                                    batch_size,
                                     int                                        n_layers,
                                     int                                        hidden_dim,
                                     int                                        quant_type,
                                     const ggml_sycl_unified::XMXConfig & xmx_config) {
    // Check for persistent TG kernel eligibility:
    // - Decode phase (token generation, not prompt processing)
    // - Batch size == 1 (single token generation)
    // - Hardware and model support (via can_use_persistent_tg)
    // - Environment variable enabled
    if (is_decode_phase && batch_size == 1 &&
        can_use_persistent_tg(n_layers, hidden_dim, quant_type, xmx_config) &&
        env_persistent_tg_enabled()) {
        return KernelType::PERSISTENT_TG;
    }

    // Fall back to legacy kernel selection
    // (Unified kernel selection happens elsewhere in the existing dispatch)
    return KernelType::LEGACY;
}

// =============================================================================
// Batch Bucket Helper
// =============================================================================

/**
 * Map batch size (M dimension) to a batch bucket for tuning key lookup.
 *
 * @param M Output rows (batch size)
 * @return BatchBucket enum value
 */
inline ggml_sycl_tuning::BatchBucket get_batch_bucket(int64_t M) {
    static int min_m = -1;
    if (min_m < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_TUNING_MIN_M");
        min_m = (env && std::atoi(env) > 0) ? std::atoi(env) : 8;
        if (min_m < 1) {
            min_m = 1;
        }
        if (min_m > 1 && get_debug_level() > 0) {
            fprintf(stderr, "[unified-dispatch] tuning min M override: %d\n", min_m);
            fflush(stderr);
        }
    }
    const int64_t tuned_m = (min_m > 1 && M < min_m) ? min_m : M;
    return ggml_sycl_tuning::bucket_for_batch(static_cast<int>(tuned_m));
}

// =============================================================================
// Extended TunedParams with Confidence
// =============================================================================

/**
 * Extended tuned params that includes confidence from the tuning engine.
 * Used for dispatch tracing and decision making.
 */
struct TunedParamsWithConfidence {
    ggml_sycl_tuning::TunedParams params;
    double confidence = 0.0;
};

// =============================================================================
// Debug Tracing
// =============================================================================

/**
 * Trace dispatch decision to stderr.
 *
 * Output format:
 * [mul_mat] M=<M> N=<N> K=<K> type=<type> -> tile=<tm>x<tn> xmx=<yes/no> conf=<pct>% src=<source>
 *
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param params Tuned parameters
 * @param confidence Confidence score (0.0 to 1.0)
 * @param source Source of params ("CACHED" or "HEURISTIC")
 */
inline void trace_dispatch(int64_t M, int64_t N, int64_t K,
                           ggml_type weight_type,
                           const ggml_sycl_tuning::TunedParams& params,
                           double confidence,
                           const char* source) {
    if (get_debug_level() >= 1) {
        fprintf(stderr, "[mul_mat] M=%lld N=%lld K=%lld type=%d -> tile=%dx%d xmx=%s conf=%d%% src=%s\n",
                static_cast<long long>(M),
                static_cast<long long>(N),
                static_cast<long long>(K),
                static_cast<int>(weight_type),
                static_cast<int>(params.tile_m),
                static_cast<int>(params.tile_n),
                params.use_dpas ? "yes" : "no",
                static_cast<int>(confidence * 100),
                source);
    }
}

// =============================================================================
// Kernel Args Builder
// =============================================================================

/**
 * Build UnifiedKernelArgs from dimensions and tuned parameters.
 *
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param params Tuned parameters
 * @param weights Quantized weight data (device pointer)
 * @param activations Activation data (device pointer, F32)
 * @param output Output data (device pointer, F32)
 * @return Populated UnifiedKernelArgs struct
 */
inline ggml_sycl_unified::UnifiedKernelArgs build_kernel_args(
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type,
    const ggml_sycl_tuning::TunedParams& params,
    const void* weights,
    const float* activations,
    float* output)
{
    ggml_sycl_unified::UnifiedKernelArgs args;
    args.M = M;
    args.N = N;
    args.K = K;
    args.tile_m = params.tile_m;
    args.tile_n = params.tile_n;
    args.tile_k = params.tile_k;
    args.use_xmx = params.use_dpas;
    args.layout_mode = params.layout_mode;
    args.layout = static_cast<ggml_sycl_unified::LayoutMode>(params.layout_mode);
    args.quant_type = static_cast<int>(weight_type);
    args.prefetch_depth = params.prefetch_depth;
    args.weights = weights;
    args.activations = activations;
    args.output = output;

    static int force_xmx = -1;
    if (force_xmx < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_FORCE_XMX");
        force_xmx = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    if (force_xmx || ggml_sycl_unified::env_force_esimd()) {
        args.use_xmx = true;
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[unified-dispatch] forcing XMX path (GGML_SYCL_UNIFIED_FORCE_XMX or GGML_SYCL_FORCE_ESIMD)\n");
            fflush(stderr);
        }
    }
    return args;
}

// =============================================================================
// Simplified Operation Context
// =============================================================================

/**
 * Simplified operation context for dispatch.
 *
 * This is a lightweight struct that doesn't require GGML tensors,
 * making it suitable for unit testing and standalone dispatch.
 */
struct OperationContext {
    int64_t   M;                // Output rows (batch * tokens)
    int64_t   N;                // Output columns (hidden dim)
    int64_t   K;                // Reduction dimension
    ggml_type weight_type;      // Quantization type
    ggml_type activation_type;  // Activation type (typically F32)
    uint32_t  device_id;        // GPU identifier

    /**
     * Build an operation context from raw dimensions.
     *
     * @param M Output rows
     * @param N Output columns
     * @param K Reduction dimension
     * @param weight_type Quantization type
     * @param activation_type Activation type
     * @param device_id GPU device ID
     * @return Populated OperationContext
     */
    static OperationContext build(int64_t M, int64_t N, int64_t K,
                                   ggml_type weight_type,
                                   ggml_type activation_type,
                                   uint32_t device_id) {
        return OperationContext{
            .M = M,
            .N = N,
            .K = K,
            .weight_type = weight_type,
            .activation_type = activation_type,
            .device_id = device_id
        };
    }
};

// =============================================================================
// Main Unified Dispatch Function (~20 lines of core logic)
// =============================================================================

/**
 * Unified matmul dispatch function.
 *
 * This is the primary entry point for dispatching quantized matrix multiplications
 * through the unified kernel architecture. It:
 *
 * 1. Builds operation context from dimensions
 * 2. Gets tuned parameters (cached or heuristic)
 * 3. Traces the dispatch decision (if debug enabled)
 * 4. Builds kernel args and launches unified matmul
 * 5. Records observation for future tuning
 *
 * Core logic is ~20 lines, replacing 200+ line legacy dispatch.
 *
 * @tparam TuningEngine Type of tuning engine (for testing with mocks)
 * @param queue SYCL queue for kernel submission
 * @param tuning_engine Reference to tuning engine for params/observation
 * @param src0_data Quantized weight data (device pointer)
 * @param src1_data Activation data (device pointer, F32)
 * @param dst_data Output data (device pointer, F32)
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 * @param device_id GPU device identifier (default: 0)
 */
template<typename TuningEngineT>
inline void ggml_sycl_mul_mat_unified(
    sycl::queue& queue,
    TuningEngineT& tuning_engine,
    const void* src0_data,      // Weights (quantized)
    const float* src1_data,     // Activations
    float* dst_data,            // Output
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type,
    ggml_sycl_unified::LayoutMode data_layout = ggml_sycl_unified::LayoutMode::AOS,
    uint32_t device_id = 0)
{
    // 1. Build tuning key for cache lookup
    ggml_sycl_tuning::TuningKey key{
        static_cast<int32_t>(weight_type),
        get_batch_bucket(M),
        static_cast<int32_t>(K),
        static_cast<int32_t>(N)
    };

    // 2. Get tuned parameters (handles cold-start internally)
    ggml_sycl_tuning::TunedParams params = tuning_engine.get_params(key);
    double confidence = tuning_engine.get_confidence(key);

    // Optional tile overrides for debugging unified-kernel correctness.
    auto parse_tile_override = [](const char * name) -> int {
        const char * env = std::getenv(name);
        if (!env || env[0] == '\0') {
            return -1;
        }
        const int v = std::atoi(env);
        return v > 0 ? v : -1;
    };
    static int override_tile_m = -2;
    static int override_tile_n = -2;
    static int override_tile_k = -2;
    if (override_tile_m == -2) override_tile_m = parse_tile_override("GGML_SYCL_UNIFIED_TILE_M");
    if (override_tile_n == -2) override_tile_n = parse_tile_override("GGML_SYCL_UNIFIED_TILE_N");
    if (override_tile_k == -2) override_tile_k = parse_tile_override("GGML_SYCL_UNIFIED_TILE_K");
    const bool has_override = (override_tile_m > 0) || (override_tile_n > 0) || (override_tile_k > 0);
    if (has_override) {
        if (override_tile_m > 0) params.tile_m = static_cast<uint16_t>(override_tile_m);
        if (override_tile_n > 0) params.tile_n = static_cast<uint16_t>(override_tile_n);
        if (override_tile_k > 0) params.tile_k = static_cast<uint16_t>(override_tile_k);
        static bool logged_override = false;
        if (!logged_override) {
            logged_override = true;
            fprintf(stderr,
                    "[unified-dispatch] tile override: M=%d N=%d K=%d\n",
                    override_tile_m, override_tile_n, override_tile_k);
            fflush(stderr);
        }
    }

    // 3. Trace if debugging enabled
    const char* source = (confidence > 0.5) ? "CACHED" : "HEURISTIC";
    trace_dispatch(M, N, K, weight_type, params, confidence, source);

    // 4. Build kernel args and launch
    // Override tuning engine layout with actual data layout
    auto args = build_kernel_args(M, N, K, weight_type, params, src0_data, src1_data, dst_data);
    args.layout = data_layout;
    args.layout_mode = static_cast<int>(data_layout);
    ggml_sycl_unified::launch_unified_matmul(queue, args);

    // 5. Record observation for tuning (non-blocking)
    tuning_engine.record_observation_async(key, params);

    // Suppress unused parameter warning for device_id (used for future multi-GPU)
    (void)device_id;
}

/**
 * Convenience wrapper for unified dispatch with default tuning engine.
 *
 * Creates a static TuningEngine instance and forwards to the template version.
 * Useful for integration points that don't manage their own tuning engine.
 *
 * @param queue SYCL queue
 * @param src0_data Quantized weight data
 * @param src1_data Activation data
 * @param dst_data Output data
 * @param M Output rows
 * @param N Output columns
 * @param K Reduction dimension
 * @param weight_type Quantization type
 */
inline void ggml_sycl_mul_mat_unified_default(
    sycl::queue& queue,
    const void* src0_data,
    const float* src1_data,
    float* dst_data,
    int64_t M, int64_t N, int64_t K,
    ggml_type weight_type,
    ggml_sycl_unified::LayoutMode data_layout = ggml_sycl_unified::LayoutMode::AOS)
{
    // Thread-safe static initialization of default tuning engine
    static ggml_sycl_tuning::TuningEngine default_engine;

    ggml_sycl_mul_mat_unified(queue, default_engine,
                               src0_data, src1_data, dst_data,
                               M, N, K, weight_type, data_layout, 0);
}

}  // namespace ggml_sycl
