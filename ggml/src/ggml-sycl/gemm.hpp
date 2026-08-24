//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_GEMM_HPP
#define GGML_SYCL_GEMM_HPP

#include "common.hpp"
#include "ggml-sycl.h"

#if GGML_SYCL_DNNL

#    include "dnnl.hpp"
#    include "dnnl_sycl.hpp"
#    include "sycl-kernel-profiler.hpp"

#    include <array>
#    include <atomic>
#    include <chrono>
#    include <cstdio>
#    include <mutex>
#    include <unordered_map>
#    include <utility>

extern int g_ggml_sycl_debug;

// llama.cpp-iikr (unphased-host-time cycle): team-lead's bisection target --
// the WOQ 2-D fallback loop below (per-expert/per-batch-element) constructs a
// fresh std::unordered_map<int, dnnl::memory> every iteration (sr83 quality
// review flagged this as a heap allocation, never quantified). Declared here,
// not in ggml-sycl.cpp's mxfp4_pp_batched_profile_accum, because this header
// is included before that struct is defined in the same translation unit --
// referencing it here would not compile. Unconditional measurement (two cheap
// chrono calls per loop iteration, negligible next to a map construction) so
// this header carries no dependency on ggml-sycl.cpp's profiling-enabled
// check; ggml-sycl.cpp's print_and_reset drains these via exchange(0) and
// folds them into its own report only when ITS OWN gate is on.
namespace ggml_sycl_gemm_profile {
// Nanoseconds, not microseconds-as-double: std::atomic<double>::fetch_add is
// a C++20 addition and this TU does not build with it. An integral atomic
// has no such restriction at any standard version this codebase targets.
inline std::atomic<int64_t> & args_map_ns_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}

inline std::atomic<int64_t> & args_map_calls_accum() {
    static std::atomic<int64_t> v{ 0 };
    return v;
}
}  // namespace ggml_sycl_gemm_profile

// =============================================================================
// oneDNN Primitive Cache
// =============================================================================
// Caches oneDNN matmul primitives to avoid JIT compilation during SYCL graph
// recording. Primitive creation involves JIT which is incompatible with graph
// recording, but execute() on a pre-created primitive is graph-compatible.
//
// Usage:
// 1. During warmup (first inference), primitives are created and cached
// 2. During graph recording, cached primitives are reused (no JIT)
// 3. Cache key includes all parameters that affect primitive creation
// =============================================================================

struct DnnlPrimitiveKey {
    int64_t                 m, n, k;
    int64_t                 batches_a, batches_b;
    dnnl::memory::data_type at, bt, ct;
    // Strides for A
    int64_t                 stra0, stra1, stra2;
    // Strides for B
    int64_t                 strb0, strb1, strb2;
    // Strides for C
    int64_t                 strc0, strc1;
    // For batch_strided: transpose flags and alpha/beta
    bool                    trans_a, trans_b;
    float                   alpha, beta;
    int64_t                 stride_a, stride_b, stride_c;
    int                     lda, ldb, ldc;
    int                     batch_size;
    // Variant: 0 = gemm, 1 = gemm_batch_strided
    int                     variant;
    int64_t                 woq_group_size;
    int                     woq_scales_mask;
    int                     woq_zp_mask;

    bool operator==(const DnnlPrimitiveKey & other) const {
        return m == other.m && n == other.n && k == other.k && batches_a == other.batches_a &&
               batches_b == other.batches_b && at == other.at && bt == other.bt && ct == other.ct &&
               stra0 == other.stra0 && stra1 == other.stra1 && stra2 == other.stra2 && strb0 == other.strb0 &&
               strb1 == other.strb1 && strb2 == other.strb2 && strc0 == other.strc0 && strc1 == other.strc1 &&
               trans_a == other.trans_a && trans_b == other.trans_b && alpha == other.alpha && beta == other.beta &&
               stride_a == other.stride_a && stride_b == other.stride_b && stride_c == other.stride_c &&
               lda == other.lda && ldb == other.ldb && ldc == other.ldc && batch_size == other.batch_size &&
               variant == other.variant && woq_group_size == other.woq_group_size &&
               woq_scales_mask == other.woq_scales_mask && woq_zp_mask == other.woq_zp_mask;
    }
};

struct DnnlPrimitiveKeyHash {
    size_t operator()(const DnnlPrimitiveKey & k) const {
        // Simple hash combining all fields
        size_t h = std::hash<int64_t>{}(k.m);
        h ^= std::hash<int64_t>{}(k.n) << 1;
        h ^= std::hash<int64_t>{}(k.k) << 2;
        h ^= std::hash<int64_t>{}(k.batches_a) << 3;
        h ^= std::hash<int64_t>{}(k.batches_b) << 4;
        h ^= std::hash<int>{}(static_cast<int>(k.at)) << 5;
        h ^= std::hash<int>{}(static_cast<int>(k.bt)) << 6;
        h ^= std::hash<int>{}(static_cast<int>(k.ct)) << 7;
        h ^= std::hash<int64_t>{}(k.stra0 + k.stra1 + k.stra2) << 8;
        h ^= std::hash<int64_t>{}(k.strb0 + k.strb1 + k.strb2) << 9;
        h ^= std::hash<int64_t>{}(k.strc0 + k.strc1) << 10;
        h ^= std::hash<int>{}(k.variant) << 11;
        h ^= std::hash<int>{}(k.batch_size) << 12;
        h ^= std::hash<int>{}(k.ldc) << 13;
        h ^= std::hash<int64_t>{}(k.woq_group_size) << 14;
        h ^= std::hash<int>{}(k.woq_scales_mask) << 15;
        h ^= std::hash<int>{}(k.woq_zp_mask) << 16;
        return h;
    }
};

struct DnnlCachedPrimitive {
    dnnl::matmul       primitive;
    dnnl::engine       engine;  // Engine the primitive was created with
    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    dnnl::memory::desc scratchpad_md;
    size_t             scratchpad_size;
};

class DnnlPrimitiveCache {
  public:
    // Get or create a cached primitive for the given key
    // Returns nullptr if creation fails
    // Note: Primitives are bound to a specific engine. If the engine changes
    // (e.g., new context between llama-bench runs), we recreate the primitive.
    const DnnlCachedPrimitive * get_or_create(const DnnlPrimitiveKey &     key,
                                              const dnnl::engine &         eng,
                                              const dnnl::memory::desc &   a_md,
                                              const dnnl::memory::desc &   b_md,
                                              const dnnl::memory::desc &   c_md,
                                              const dnnl::primitive_attr & attr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Check if cached primitive's engine matches current engine
            // oneDNN primitives are bound to a specific engine and cannot
            // be executed on a stream from a different engine
            if (it->second.engine == eng) {
                return &it->second;
            }
            // Engine mismatch - need to recreate primitive for new engine
            cache_.erase(it);
        }

        // Create new primitive
        try {
            DnnlCachedPrimitive cached;
            cached.engine          = eng;  // Store engine for future comparisons
            auto matmul_pd         = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            cached.a_md            = matmul_pd.src_desc();
            cached.b_md            = matmul_pd.weights_desc();
            cached.c_md            = matmul_pd.dst_desc();
            cached.scratchpad_md   = matmul_pd.scratchpad_desc();
            cached.scratchpad_size = cached.scratchpad_md.get_size();
            cached.primitive       = dnnl::matmul(matmul_pd);

            auto result = cache_.emplace(key, std::move(cached));
            return &result.first->second;
        } catch (const dnnl::error & e) {
            // Failed to create primitive
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN] matmul primitive creation failed: %s\n", e.what());
            }
            return nullptr;
        }
    }

    // Check if a primitive exists for the given key
    bool has(const DnnlPrimitiveKey & key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    // Get cached primitive (returns nullptr if not found)
    const DnnlCachedPrimitive * get(const DnnlPrimitiveKey & key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = cache_.find(key);
        return it != cache_.end() ? &it->second : nullptr;
    }

    // Clear the cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    // Get cache size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    // Get maximum scratchpad size across all cached primitives
    // Used to pre-allocate scratchpad pool before graph recording
    size_t get_max_scratchpad_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t                      max_size = 0;
        for (const auto & [key, cached] : cache_) {
            max_size = std::max(max_size, cached.scratchpad_size);
        }
        return max_size;
    }

  private:
    mutable std::mutex                                                              mutex_;
    std::unordered_map<DnnlPrimitiveKey, DnnlCachedPrimitive, DnnlPrimitiveKeyHash> cache_;
};

// Global primitive cache (shared across contexts)
inline DnnlPrimitiveCache & get_dnnl_primitive_cache() {
    static DnnlPrimitiveCache cache;
    return cache;
}

class DnnlGemmWrapper {
  public:
    using dt  = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;

    // Serialize oneDNN execution to avoid cross-thread primitive/memory races.
    // CPU uses a dedicated mutex. Each GPU device gets its own mutex so parallel
    // multi-GPU dispatch (e.g. B580 + B50) does not serialize against each other.
    static std::mutex & exec_mutex_cpu() {
        static std::mutex mutex;
        return mutex;
    }

    static std::mutex & exec_mutex(const queue_ptr & q) {
        if (q == ggml_sycl_get_cpu_queue()) {
            return exec_mutex_cpu();
        }
        // Per-GPU device mutex (allows parallel B580+B50 GEMM without serialization)
        static std::array<std::mutex, GGML_SYCL_MAX_DEVICES> gpu_mutexes;
        int                                                  dev_id = ggml_sycl_get_device_id_from_queue(*q);
        return gpu_mutexes[std::min(dev_id, static_cast<int>(gpu_mutexes.size()) - 1)];
    }

    template <typename T> static constexpr dt to_dt() {
        if constexpr (std::is_same_v<T, float>) {
            return dt::f32;
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            return dt::f16;
        }
#    ifdef GGML_SYCL_HAS_BF16
        else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
            return dt::bf16;
        }
#    endif
        else {
            static_assert(0);
        }
    }

    // Byte width of the dense (non-sub-byte) dtypes woq_gemm_batch_mxfp4's
    // activation/output pointer arithmetic needs; 0 for anything else
    // (callers treat that as "unsupported" and refuse). llama.cpp-sr83
    // (quality review finding #9): was two identical local lambdas.
    static constexpr int64_t dt_size_bytes(dt t) {
        switch (t) {
            case dt::f32:
                return 4;
            case dt::f16:
                return 2;
            default:
                return 0;
        }
    }

    static void gemm(ggml_backend_sycl_context & ctx,
                     int                         m,
                     int                         n,
                     int                         k,
                     const void *                a,
                     dt                          at,
                     dnnl_dim_t                  stra0,
                     dnnl_dim_t                  stra1,
                     dnnl_dim_t                  stra2,
                     const void *                b,
                     dt                          bt,
                     dnnl_dim_t                  strb0,
                     dnnl_dim_t                  strb1,
                     dnnl_dim_t                  strb2,
                     void *                      c,
                     dt                          ct,
                     const queue_ptr &           q,
                     dnnl_dim_t                  batches_a,
                     dnnl_dim_t                  batches_b,
                     int                         ldc = -1) {
        std::lock_guard<std::mutex> lock(exec_mutex(q));

        auto stream = ctx.stream_dnnl(q);
        auto eng    = ctx.engine_dnnl(q);

        if (ldc <= 0) {
            ldc = m;
        }

        // Build cache key
        DnnlPrimitiveKey key{};
        key.m         = m;
        key.n         = n;
        key.k         = k;
        key.batches_a = batches_a;
        key.batches_b = batches_b;
        key.at        = at;
        key.bt        = bt;
        key.ct        = ct;
        key.stra0     = stra0;
        key.stra1     = stra1;
        key.stra2     = stra2;
        key.strb0     = strb0;
        key.strb1     = strb1;
        key.strb2     = strb2;
        key.ldc       = ldc;
        key.variant   = 0;  // gemm variant

        // Build memory descriptors
        dnnl::memory::dims a_dims    = { batches_a, m, k };
        dnnl::memory::dims a_strides = { stra2, stra1, stra0 };
        const auto         a_in_md   = dnnl::memory::desc(a_dims, at, a_strides);

        dnnl::memory::dims b_dims    = { batches_b, k, n };
        dnnl::memory::dims b_strides = { strb2, strb0, strb1 };
        const auto         b_in_md   = dnnl::memory::desc(b_dims, bt, b_strides);

        dnnl::memory::dims c_dims    = { std::max(batches_a, batches_b), m, n };
        dnnl::memory::dims c_strides = { static_cast<dnnl_dim_t>(ldc) * n, 1, ldc };
        const auto         c_md      = dnnl::memory::desc(c_dims, ct, c_strides);

        dnnl::primitive_attr primitive_attr;
        primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#    ifdef GGML_SYCL_F16
        primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#    endif

        // Get or create cached primitive
        auto &                      cache  = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive * cached = cache.get_or_create(key, eng, a_in_md, b_in_md, c_md, primitive_attr);

        if (!cached) {
            // Fallback: create primitive directly if caching fails
            auto         a_mem           = dnnl::memory(a_in_md, eng, const_cast<void *>(a));
            auto         b_mem           = dnnl::memory(b_in_md, eng, const_cast<void *>(b));
            auto         matmul_pd       = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
            auto         c_mem           = dnnl::memory(matmul_pd.dst_desc(), eng, c);
            auto         scratchpad_md   = matmul_pd.scratchpad_desc();
            const size_t scratchpad_size = scratchpad_md.get_size();
            auto         matmul_prim     = dnnl::matmul(matmul_pd);

            std::unordered_map<int, dnnl::memory> matmul_args;
            matmul_args.insert({ DNNL_ARG_SRC, a_mem });
            matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
            matmul_args.insert({ DNNL_ARG_DST, c_mem });
            if (scratchpad_size > 0) {
                auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);
                if (scratchpad_mem.get(true) == nullptr) {
                    throw std::runtime_error("oneDNN scratchpad allocation failed");
                }
                matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
            }
            matmul_prim.execute(stream, matmul_args);
            return;
        }

        // Use cached primitive - only memory binding and execute (graph-compatible)
        auto         a_mem           = dnnl::memory(cached->a_md, eng, const_cast<void *>(a));
        auto         b_mem           = dnnl::memory(cached->b_md, eng, const_cast<void *>(b));
        auto         c_mem           = dnnl::memory(cached->c_md, eng, c);
        const size_t scratchpad_size = cached->scratchpad_md.get_size();

        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({ DNNL_ARG_SRC, a_mem });
        matmul_args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        matmul_args.insert({ DNNL_ARG_DST, c_mem });
        if (scratchpad_size > 0) {
            auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);
            if (scratchpad_mem.get(true) == nullptr) {
                throw std::runtime_error("oneDNN scratchpad allocation failed");
            }
            matmul_args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
        }

        cached->primitive.execute(stream, matmul_args);
    }

    static void row_gemm(ggml_backend_sycl_context & ctx,
                         int                         m,
                         int                         n,
                         int                         k,
                         const void *                a,
                         dt                          at,
                         const void *                b,
                         dt                          bt,
                         void *                      c,
                         dt                          ct,
                         const queue_ptr &           q,
                         int                         ldc = -1) {
        gemm(ctx, m, n, k, a, at, 1, k, k * m, b, bt, 1, k, n * k, c, ct, q, 1, 1, ldc);
    }

    // WoQ GEMM for Q4_0 weights (s4) with grouped scales/zero-points.
    // A: [m, k] row-major, B: [k, n] row-major (s4), C: [m, n] row-major.
    static bool woq_gemm_q4_0(ggml_backend_sycl_context & ctx,
                              int                         m,
                              int                         n,
                              int                         k,
                              const void *                a,
                              dt                          at,
                              const void *                b_s4,
                              int64_t                     group_size,
                              const float *               scales,
                              const int8_t *              zero_points,
                              void *                      c,
                              dt                          ct,
                              const queue_ptr &           q,
                              int64_t                     c_stride0,
                              int64_t                     c_stride1) {
        return woq_gemm_q4_0_impl(ctx, m, n, k, a, at, b_s4, /* b_bytes = */ 0, /* b_is_packed = */ false, group_size,
                                  scales, zero_points, c, ct, q, c_stride0, c_stride1);
    }

    // WoQ GEMM with pre-packed oneDNN weights (b_packed uses cached->b_md layout).
    static bool woq_gemm_q4_0_packed(ggml_backend_sycl_context & ctx,
                                     int                         m,
                                     int                         n,
                                     int                         k,
                                     const void *                a,
                                     dt                          at,
                                     const void *                b_packed,
                                     size_t                      b_packed_bytes,
                                     int64_t                     group_size,
                                     const float *               scales,
                                     const int8_t *              zero_points,
                                     void *                      c,
                                     dt                          ct,
                                     const queue_ptr &           q,
                                     int64_t                     c_stride0,
                                     int64_t                     c_stride1) {
        return woq_gemm_q4_0_impl(ctx, m, n, k, a, at, b_packed, b_packed_bytes, /* b_is_packed = */ true, group_size,
                                  scales, zero_points, c, ct, q, c_stride0, c_stride1);
    }

  private:
    // The proven WOQ-MXFP4 (f4_e2m1 nibbles + e8m0 grouped scales) 2-D
    // scale-mask/group_dims recipe -- C1's exact spike result (llama.cpp-4m9p),
    // and (post-sr83) the correctness-verified default arm of
    // woq_gemm_batch_mxfp4's fallback below. Factored out so there is ONE
    // place that builds this attr set: both the real 2-D GEMM path and the
    // E1 capability probe (woq_mxfp4_supported) call it, instead of carrying
    // two copies that could silently drift apart. Behavior-preserving
    // extraction only -- same mask, same group_dims shape, same fpmath call.
    static constexpr int woq_mxfp4_2d_scale_mask = (1 << 0) | (1 << 1);

    static dnnl::primitive_attr woq_mxfp4_2d_attr(int64_t group_size) {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        dnnl::memory::dims group_dims = { group_size, 1 };
        attr.set_scales(DNNL_ARG_WEIGHTS, woq_mxfp4_2d_scale_mask, group_dims, dt::e8m0);
#    ifdef GGML_SYCL_F16
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);
#    endif
        return attr;
    }

    static bool woq_gemm_q4_0_impl(ggml_backend_sycl_context & ctx,
                                   int                         m,
                                   int                         n,
                                   int                         k,
                                   const void *                a,
                                   dt                          at,
                                   const void *                b_data,
                                   size_t                      b_bytes,
                                   bool                        b_is_packed,
                                   int64_t                     group_size,
                                   const float *               scales,
                                   const int8_t *              zero_points,
                                   void *                      c,
                                   dt                          ct,
                                   const queue_ptr &           q,
                                   int64_t                     c_stride0,
                                   int64_t                     c_stride1) {
        if (!a || !b_data || !scales || !zero_points || !c) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] null pointer(s) provided\n");
            }
            return false;
        }
        if (m <= 0 || n <= 0 || k <= 0 || group_size <= 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] invalid dims m=%d n=%d k=%d group=%lld\n", m, n, k,
                             static_cast<long long>(group_size));
            }
            return false;
        }
        if (c_stride0 <= 0 || c_stride1 <= 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] invalid C strides %lld,%lld\n", static_cast<long long>(c_stride0),
                             static_cast<long long>(c_stride1));
            }
            return false;
        }
        if ((k % group_size) != 0) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] K not divisible by group size (k=%d group=%lld)\n", k,
                             static_cast<long long>(group_size));
            }
            return false;
        }

        const int64_t groups = k / group_size;

        std::lock_guard<std::mutex> lock(exec_mutex(q));
        auto                        stream = ctx.stream_dnnl(q);
        auto                        eng    = ctx.engine_dnnl(q);

        const dnnl::memory::desc a_md({ m, k }, at, { k, 1 });
        const dnnl::memory::desc b_user_md({ k, n }, dt::s4, { n, 1 });
        const dnnl::memory::desc c_md({ m, n }, ct, { c_stride0, c_stride1 });
        const dnnl::memory::desc b_any_md({ k, n }, dt::s4, tag::any);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const int   mask       = (1 << 0) | (1 << 1);
        dnnl_dims_t group_dims = { group_size, 1 };
        if (dnnl_primitive_attr_set_scales(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                           dnnl::memory::convert_to_c(dt::f32)) != dnnl_success) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] set_scales failed\n");
            }
            return false;
        }
        if (dnnl_primitive_attr_set_zero_points(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                                dnnl::memory::convert_to_c(dt::s8)) != dnnl_success) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] set_zero_points failed\n");
            }
            return false;
        }
#    ifdef GGML_SYCL_F16
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);
#    endif

        DnnlPrimitiveKey key{};
        key.m               = m;
        key.n               = n;
        key.k               = k;
        key.batches_a       = 1;
        key.batches_b       = 1;
        key.at              = at;
        key.bt              = dt::s4;
        key.ct              = ct;
        key.stra0           = 1;
        key.stra1           = k;
        key.stra2           = static_cast<int64_t>(m) * k;
        key.strb0           = 1;
        key.strb1           = n;
        key.strb2           = static_cast<int64_t>(k) * n;
        key.ldc             = n;
        key.variant         = 2;
        key.strc0           = c_stride0;
        key.strc1           = c_stride1;
        key.woq_group_size  = group_size;
        key.woq_scales_mask = mask;
        key.woq_zp_mask     = mask;

        auto &                      cache  = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive * cached = cache.get_or_create(key, eng, a_md, b_any_md, c_md, attr);
        if (!cached) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ] primitive cache miss+create failed\n");
            }
            return false;
        }

        auto a_mem = dnnl::memory(cached->a_md, eng, const_cast<void *>(a));
        auto c_mem = dnnl::memory(cached->c_md, eng, c);

        dnnl::memory          b_mem        = {};
        void *                b_packed_dev = nullptr;
        ggml_sycl::mem_handle b_packed_owner;
        if (b_is_packed) {
            const size_t packed_bytes = cached->b_md.get_size();
            if (b_bytes > 0 && b_bytes < packed_bytes) {
                if (g_ggml_sycl_debug) {
                    std::fprintf(stderr, "[ONEDNN][WOQ] packed weights too small (%zu < %zu)\n", b_bytes, packed_bytes);
                }
                return false;
            }
            b_mem = dnnl::memory(cached->b_md, eng, const_cast<void *>(b_data));
        } else {
            dnnl::memory b_user_mem(b_user_md, eng, const_cast<void *>(b_data));
            b_mem = b_user_mem;
            if (cached->b_md != b_user_mem.get_desc()) {
                const size_t             packed_bytes = cached->b_md.get_size();
                ggml_sycl::alloc_request req{};
                req.queue                          = q;
                req.device                         = ggml_sycl_get_device_id_from_queue(*q);
                req.size                           = packed_bytes;
                req.intent.role                    = ggml_sycl::alloc_role::STAGING;
                req.intent.category                = ggml_sycl::runtime_category::STAGING;
                req.intent.cohort_id               = "onednn_woq_packed";
                req.intent.constraints.must_device = true;

                ggml_sycl::alloc_handle b_packed_alloc_owner{};
                if (ggml_sycl::unified_alloc(req, &b_packed_alloc_owner) && b_packed_alloc_owner.ptr) {
                    b_packed_owner =
                        ggml_sycl::detail::from_legacy_owned_alloc(std::move(b_packed_alloc_owner), GGML_LAYOUT_AOS);
                    auto resolved = b_packed_owner.resolve(req.device);
                    b_packed_dev  = resolved && resolved.on_device ? resolved.ptr : nullptr;
                    if (!b_packed_dev) {
                        b_packed_owner = {};
                    }
                }
                if (!b_packed_dev) {
                    if (g_ggml_sycl_debug) {
                        std::fprintf(stderr, "[ONEDNN][WOQ] packed weights alloc failed (%zu bytes)\n", packed_bytes);
                    }
                    return false;
                }
                b_mem = dnnl::memory(cached->b_md, eng, b_packed_dev);
                dnnl::reorder(b_user_mem, b_mem).execute(stream, b_user_mem, b_mem);
                stream.wait();
            }
        }

        dnnl::memory scales_mem(
            {
                { n, groups },
                dt::f32, { 1, n      }
        },
            eng, const_cast<float *>(scales));
        dnnl::memory zp_mem(
            {
                { n, groups },
                dt::s8, { 1, n      }
        },
            eng, const_cast<int8_t *>(zero_points));

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, a_mem });
        args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        args.insert({ DNNL_ARG_DST, c_mem });
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem });
        args.insert({ DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem });
        if (cached->scratchpad_size > 0) {
            auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);
            if (scratchpad_mem.get(true) == nullptr) {
                throw std::runtime_error("oneDNN scratchpad allocation failed");
            }
            args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
        }

        cached->primitive.execute(stream, args);

        if (b_packed_dev) {
            stream.wait();
            b_packed_owner = {};
        }
        return true;
    }

  public:
    // Strided batch GEMM - C[i] = alpha * A[i] * B[i] + beta * C[i]
    // Matches dpct::gemm_batch interface for strided buffers
    //
    // `deps` are SYCL events the GEMM must wait on before executing (e.g. the
    // dequant/activation staging that fills `a`/`b`). The returned event fires
    // when the GEMM completes, so callers on the same in-order queue can chain
    // dependent submissions onto it (e.g. via cgh.depends_on(...) or a barrier)
    // instead of relying on in-order-queue ordering with an async primitive.
    static sycl::event gemm_batch_strided(ggml_backend_sycl_context &      ctx,
                                          bool                             trans_a,
                                          bool                             trans_b,
                                          int                              m,
                                          int                              n,
                                          int                              k,
                                          float                            alpha,
                                          const void *                     a,
                                          dt                               at,
                                          int                              lda,
                                          int64_t                          stride_a,
                                          const void *                     b,
                                          dt                               bt,
                                          int                              ldb,
                                          int64_t                          stride_b,
                                          float                            beta,
                                          void *                           c,
                                          dt                               ct,
                                          int                              ldc,
                                          int64_t                          stride_c,
                                          int                              batch_size,
                                          const queue_ptr &                q,
                                          const std::vector<sycl::event> & deps = {}) {
        std::lock_guard<std::mutex> lock(exec_mutex(q));
        auto                        stream = ctx.stream_dnnl(q);
        auto                        eng    = ctx.engine_dnnl(q);

        // Build cache key for batch_strided variant
        DnnlPrimitiveKey key{};
        key.m          = m;
        key.n          = n;
        key.k          = k;
        key.at         = at;
        key.bt         = bt;
        key.ct         = ct;
        key.trans_a    = trans_a;
        key.trans_b    = trans_b;
        key.alpha      = alpha;
        key.beta       = beta;
        key.stride_a   = stride_a;
        key.stride_b   = stride_b;
        key.stride_c   = stride_c;
        key.lda        = lda;
        key.ldb        = ldb;
        key.ldc        = ldc;
        key.batch_size = batch_size;
        key.variant    = 1;  // gemm_batch_strided variant

        // oneDNN matmul: C = A * B where A is (batch, M, K), B is (batch, K, N),
        // C is (batch, M, N). The logical dims are fixed; BLAS-style trans_a/
        // trans_b describe how the operand is STORED (column-major with leading
        // dimension ld*, trans meaning the stored matrix is the logical one
        // transposed), so the trans flag flips which logical dim gets stride 1.
        //
        // llama.cpp-dboi: the previous encoding swapped the DIMS for a
        // transposed operand while keeping the {1, ld} strides, which hands
        // oneDNN a column-major read of the buffer under relabeled M/K -- for
        // square operands (GPT-OSS gate/up, 2880x2880) that builds fine and
        // silently computes with the weight matrix transposed; non-square
        // shapes would have failed primitive creation instead.
        dnnl::memory::dims a_dims = { batch_size, m, k };
        dnnl::memory::dims b_dims = { batch_size, k, n };
        dnnl::memory::dims c_dims = { batch_size, m, n };

        dnnl::memory::dims a_strides =
            trans_a ? dnnl::memory::dims{ stride_a, lda, 1 } : dnnl::memory::dims{ stride_a, 1, lda };
        dnnl::memory::dims b_strides =
            trans_b ? dnnl::memory::dims{ stride_b, ldb, 1 } : dnnl::memory::dims{ stride_b, 1, ldb };
        dnnl::memory::dims c_strides = { stride_c, 1, ldc };

        const auto a_md = dnnl::memory::desc(a_dims, at, a_strides);
        const auto b_md = dnnl::memory::desc(b_dims, bt, b_strides);
        const auto c_md = dnnl::memory::desc(c_dims, ct, c_strides);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        // Handle alpha and beta via post-ops if not 1.0/0.0
        if (alpha != 1.0f || beta != 0.0f) {
            dnnl::post_ops po;
            if (beta != 0.0f) {
                // C = alpha * (A * B) + beta * C
                // oneDNN does: dst = src * alpha + dst * beta with sum post-op
                po.append_sum(beta);
            }
            if (alpha != 1.0f) {
                po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
            }
            attr.set_post_ops(po);
        }

#    ifdef GGML_SYCL_F16
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
#    endif

        // Get or create cached primitive
        auto &                      cache  = get_dnnl_primitive_cache();
        const DnnlCachedPrimitive * cached = cache.get_or_create(key, eng, a_md, b_md, c_md, attr);

        if (!cached) {
            // Fallback: create primitive directly if caching fails
            auto a_mem          = dnnl::memory(a_md, eng, const_cast<void *>(a));
            auto b_mem          = dnnl::memory(b_md, eng, const_cast<void *>(b));
            auto matmul_pd      = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            auto c_mem          = dnnl::memory(matmul_pd.dst_desc(), eng, c);
            auto scratchpad_md  = matmul_pd.scratchpad_desc();
            auto scratchpad_mem = ctx.get_scratchpad_mem(scratchpad_md, eng, q);
            if (scratchpad_mem.get(true) == nullptr && scratchpad_md.get_size() > 0) {
                throw std::runtime_error("oneDNN scratchpad allocation failed");
            }
            auto matmul_prim = dnnl::matmul(matmul_pd);

            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, a_mem });
            args.insert({ DNNL_ARG_WEIGHTS, b_mem });
            args.insert({ DNNL_ARG_DST, c_mem });
            if (scratchpad_md.get_size() > 0) {
                args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
            }
            return dnnl::sycl_interop::execute(matmul_prim, stream, args, deps);
        }

        // Use cached primitive - only memory binding and execute (graph-compatible)
        auto a_mem = dnnl::memory(cached->a_md, eng, const_cast<void *>(a));
        auto b_mem = dnnl::memory(cached->b_md, eng, const_cast<void *>(b));
        auto c_mem = dnnl::memory(cached->c_md, eng, c);

        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, a_mem });
        args.insert({ DNNL_ARG_WEIGHTS, b_mem });
        args.insert({ DNNL_ARG_DST, c_mem });
        // llama.cpp-dboi: a zero-size scratchpad desc makes get_scratchpad_mem()
        // return a default-constructed dnnl::memory, and passing that
        // uninitialized object as an arg makes sycl_interop::execute throw
        // "object is not initialized". Guard the insert on the size, exactly
        // as the gemm (variant 0) branches above already do.
        if (cached->scratchpad_md.get_size() > 0) {
            auto scratchpad_mem = ctx.get_scratchpad_mem(cached->scratchpad_md, eng, q);
            if (scratchpad_mem.get(true) == nullptr) {
                throw std::runtime_error("oneDNN scratchpad allocation failed");
            }
            args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
        }

        return dnnl::sycl_interop::execute(cached->primitive, stream, args, deps);
    }

    // Per-device capability probe for the WOQ-MXFP4 (f4_e2m1 nibbles + e8m0
    // grouped scales) GEMM path, following the same philosophy the C1 spike
    // (llama.cpp-4m9p) established for this GEMM family: the ATTEMPT to
    // build C1's exact proven 2-D primitive_desc recipe (woq_mxfp4_2d_attr,
    // above) IS the capability query. Any dnnl::error thrown during that
    // attempt means "not supported on this device's oneDNN build" and is
    // caught here, never propagated.
    //
    // Uses fixed tiny probe dims per the epic plan (docs/plans/2026-08-21-
    // gptoss-perf-recovery-epic.md, Task E1: probe_m=8, probe_n=128),
    // independent of any real GEMM shape -- primitive_desc creation either
    // accepts the (f4_e2m1, e8m0) dtype/attr combination for this engine or
    // it doesn't; that answer does not depend on m/n. group_size is
    // QK_MXFP4 -- the exact value the real WOQ-MXFP4 route uses, not an
    // arbitrary probe-only choice: the probe has to ask the exact question
    // the route asks, since a build that accepts one group_size but refuses
    // another would otherwise make the probe wrong in whichever direction.
    // probe_k=64 keeps `k % group_size == 0` (two groups). The dtype pair
    // probed (f16 activations x f4_e2m1 weights -> f32 output) matches
    // exactly what the shipped WOQ-MXFP4 caller passes -- the probe answers
    // for that specific combination, not WOQ-MXFP4 in the abstract.
    //
    // Result is cached per device, keyed the same way exec_mutex(q) keys its
    // per-GPU mutexes (device id from the queue, clamped into
    // GGML_SYCL_MAX_DEVICES) and using the same tri-state atomic idiom
    // woq_gemm_batch_mxfp4 uses for its own per-device arm cache below --
    // but the two caches answer DIFFERENT questions: this cache's -1 means
    // the WOQ-MXFP4 route is unavailable on the device at all, while
    // woq_gemm_batch_mxfp4's `tri_state` -1 means only its opt-in 3-D
    // grouped-batch arm is unavailable (its 2-D per-batch fallback still
    // runs). 0 = not yet probed, 1 = supported, -1 = unsupported. Once set,
    // later calls return the cached answer without touching oneDNN again --
    // zero cost after the first call per device.
    //
    // Precondition: must NOT be called with exec_mutex(q) already held --
    // the lock below is non-recursive, so a caller already holding it would
    // deadlock on itself.
    static bool woq_mxfp4_supported(ggml_backend_sycl_context & ctx, const queue_ptr & q) {
        static std::array<std::atomic<int>, GGML_SYCL_MAX_DEVICES> woq_mxfp4_supported_cache;

        const int dev_id =
            std::min(ggml_sycl_get_device_id_from_queue(*q), static_cast<int>(woq_mxfp4_supported_cache.size()) - 1);

        int cached = woq_mxfp4_supported_cache[dev_id].load(std::memory_order_acquire);
        if (cached != 0) {
            return cached > 0;
        }

        std::lock_guard<std::mutex> lock(exec_mutex(q));
        // Re-check under the lock: another thread may have already probed
        // this device while we were waiting.
        cached = woq_mxfp4_supported_cache[dev_id].load(std::memory_order_acquire);
        if (cached != 0) {
            return cached > 0;
        }

        // llama.cpp-t9x5 (spec review fix 1): everything that can throw a
        // dnnl::error for an unsupported (f4_e2m1, e8m0) combination --
        // engine_dnnl, the memory::desc constructions, and
        // woq_mxfp4_2d_attr's set_scales/set_fpmath_mode C-API calls -- must
        // live INSIDE the try. dtype/attr validation happens at those
        // construction points, not only at primitive_desc creation, so
        // leaving any of this outside the try let the exact failure the
        // probe exists to convert to `false` escape uncaught.
        bool supported = false;
        try {
            auto eng = ctx.engine_dnnl(q);

            constexpr int64_t probe_m          = 8;
            constexpr int64_t probe_k          = 64;
            constexpr int64_t probe_n          = 128;
            constexpr int64_t probe_group_size = QK_MXFP4;

            const dnnl::memory::desc   a_md({ probe_m, probe_k }, dt::f16, { probe_k, 1 });
            const dnnl::memory::desc   b_md({ probe_k, probe_n }, dt::f4_e2m1, { probe_n, 1 });
            const dnnl::memory::desc   c_md({ probe_m, probe_n }, dt::f32, { probe_n, 1 });
            const dnnl::primitive_attr attr = woq_mxfp4_2d_attr(probe_group_size);

            dnnl::matmul::primitive_desc probe_pd(eng, a_md, b_md, c_md, attr);
            (void) probe_pd;
            supported = true;
            // N3 (spec review): a once-per-device POSITIVE control -- without
            // this, "supported" and "never probed" are indistinguishable in
            // the log. No separate arm_print_latch is needed here (unlike
            // woq_gemm_batch_mxfp4's below): this try body itself only ever
            // runs once per device, gated by the double-checked cache read
            // above, so this line cannot double-print.
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4] capability probe accepted device=%d\n", dev_id);
            }
        } catch (const dnnl::error & e) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4] capability probe refused: %s\n", e.what());
            }
        } catch (const std::exception & e) {
            // llama.cpp-t9x5 (spec review fix 4): engine_dnnl() and the
            // memory::desc/attr constructions above can also throw non-dnnl
            // exceptions (e.g. std::bad_alloc, std::system_error). The
            // contract this function makes to every caller is "you get a
            // bool" -- any construction failure means "not supported", never
            // an escaping exception.
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4] capability probe refused (non-dnnl exception): %s\n",
                             e.what());
            }
        } catch (...) {
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4] capability probe refused (unknown exception)\n");
            }
        }

        woq_mxfp4_supported_cache[dev_id].store(supported ? 1 : -1, std::memory_order_release);
        return supported;
    }

    // Batched WoQ GEMM for MXFP4 weights (f4_e2m1 nibbles + e8m0 grouped
    // scales) -- the batched-PP counterpart of woq_gemm_q4_0_impl, extended
    // with a batch dimension the same way gemm_batch_strided extends gemm().
    // Row-major native throughout, matching exactly what the repack kernels
    // produce (convert.cpp: repack_mxfp4_soa_to_woq / _xmx_tiled_to_woq), so
    // no transposition sits between repack output and this GEMM's operands.
    // Unlike woq_gemm_q4_0_impl this uses FIXED memory descriptors rather
    // than tag::any -- the C1 spike (llama.cpp-4m9p) proved f4_e2m1 accepts
    // the plain strided layout directly, so there is no packed-layout
    // reorder branch to carry here:
    //   A (SRC)     = f16 activations   [batch, m, k]            strides {stride_a, k, 1}
    //   B (WEIGHTS) = f4_e2m1 nibbles   [batch, k, n]             strides {stride_b, n, 1}
    //   scales      = e8m0              [batch, k/group_size, n] strides {stride_scales, n, 1}
    //   C (DST)     = f32 output        [batch, m, n]             strides {stride_c, n, 1}
    // m = tokens (padded group rows), k = input features, n = output
    // features. stride_a/stride_c are element strides (matches
    // gemm_batch_strided's convention); stride_b/stride_scales are element
    // strides in the SAME units the weight/scale memory descriptors use
    // (nibble count k*n and scale count groups*n respectively -- oneDNN
    // divides by elements-per-byte internally for the sub-byte f4_e2m1
    // type, exactly as it already does for s4 in woq_gemm_q4_0_impl).
    //
    // The scale mask/group_dims are the C1 spike's exact proven 2-D recipe
    // (mask 3, group_dims {group_size,1}) extended to the batch axis (mask
    // 7, group_dims {1,group_size,1}) -- the batch-axis extension is
    // unvalidated on hardware. If primitive creation with the 3-D mask
    // refuses, this falls back to looping C1's proven 2-D primitive once per
    // batch element and merging the per-element completion events into one
    // returned barrier event. The per-device outcome is cached after the
    // first call so steady-state dispatch never re-probes. See the C3
    // tracker report (llama.cpp-sr83) for which arm exercised on hardware.
    //
    // `deps` are SYCL events this GEMM must wait on (repack + activation
    // staging); the returned event fires on GEMM completion, matching
    // gemm_batch_strided's deps-in/event-out contract.
    static sycl::event woq_gemm_batch_mxfp4(ggml_backend_sycl_context &      ctx,
                                            int                              m,
                                            int                              n,
                                            int                              k,
                                            const void *                     a,
                                            dt                               at,
                                            int64_t                          stride_a,
                                            const void *                     b_nibbles,
                                            int64_t                          stride_b,
                                            int64_t                          group_size,
                                            const void *                     b_scales,
                                            int64_t                          stride_scales,
                                            void *                           c,
                                            dt                               ct,
                                            int64_t                          stride_c,
                                            int                              batch_size,
                                            const queue_ptr &                q,
                                            const std::vector<sycl::event> & deps = {}) {
        if (m <= 0 || n <= 0 || k <= 0 || batch_size <= 0 || group_size <= 0 || (k % group_size) != 0) {
            throw std::runtime_error("woq_gemm_batch_mxfp4: invalid dims/group_size");
        }
        const int64_t elem_bytes     = dt_size_bytes(at);
        const int64_t out_elem_bytes = dt_size_bytes(ct);
        if (elem_bytes == 0 || out_elem_bytes == 0 || (stride_b % 2) != 0) {
            throw std::runtime_error("woq_gemm_batch_mxfp4: unsupported activation/output dtype or odd nibble stride");
        }
        const int64_t groups = k / group_size;

        std::lock_guard<std::mutex> lock(exec_mutex(q));
        auto                        stream = ctx.stream_dnnl(q);
        auto                        eng    = ctx.engine_dnnl(q);
        auto &                      cache  = get_dnnl_primitive_cache();

        const dnnl::memory::desc a_md({ batch_size, m, k }, at, { stride_a, k, 1 });
        const dnnl::memory::desc c_md({ batch_size, m, n }, ct, { stride_c, n, 1 });
        auto                     a_mem = dnnl::memory(a_md, eng, const_cast<void *>(a));
        auto                     c_mem = dnnl::memory(c_md, eng, c);

        // Per-device cache of whether the batch-dim-grouped 3-D scale mask
        // is accepted by this device's oneDNN build -- avoids re-probing
        // primitive creation (which internally try/catches a dnnl::error)
        // on every dispatch once the answer is known. Left default-
        // constructed (no initializer): std::atomic has no copy/move
        // constructor, so an IIFE-returned std::array<std::atomic<int>,N>
        // does not compile -- a static-storage-duration array of atomics is
        // zero-initialized before any dynamic initialization regardless, and
        // std::atomic<int>'s defaulted default constructor does nothing
        // beyond that, so this reliably starts at 0.
        // 0 = unknown (try 3-D), 1 = supported, -1 = unsupported (2-D only).
        static std::array<std::atomic<int>, GGML_SYCL_MAX_DEVICES> tri_state;
        const int dev_id = std::min(ggml_sycl_get_device_id_from_queue(*q), static_cast<int>(tri_state.size()) - 1);

        // Once-per-device latch for the arm-identity print below (separate
        // from tri_state, which caches the CAPABILITY answer and stays
        // consulted -- and re-satisfied -- on every call once known
        // supported; this only gates how many times the print fires).
        static std::array<std::atomic<int>, GGML_SYCL_MAX_DEVICES> arm_print_latch;

        // llama.cpp-sr83 (fix cycle 2, team-lead hardware verdict
        // 2026-08-22): DEFAULT FLIPPED. The discriminating experiment this
        // env was added for is decisive and reproducible on hardware (B70):
        // the batch-dim-grouped 3-D scale mask (mask 7, group_dims
        // {1,group_size,1}) is ACCEPTED by oneDNN but produces wrong
        // numerics (down role, blk.0 gpu=-5.21 vs cpu=-484.47, NaN cascade
        // by blk.1) -- deterministic and card-independent. The 2-D
        // per-batch-element loop (C1's exact proven non-batched recipe) is
        // clean on the SAME hardware: all roles, all blocks, max_rel <=
        // 0.0258, zero NaNs. Correctness-by-default: the proven recipe now
        // ships as the default, and the unproven 3-D encoding is opt-IN via
        // GGML_SYCL_MOE_PP_WOQ_3D=1 for whoever picks up the follow-up
        // encoding-rework task. Does NOT poison tri_state -- an opt-in run
        // must not overwrite the real capability cache for the next
        // default-env run.
        static const bool pp_woq_3d_enabled = []() {
            const char * env = std::getenv("GGML_SYCL_MOE_PP_WOQ_3D");
            return env != nullptr && std::atoi(env) != 0;
        }();

        const dnnl::memory::desc b_md_3d({ batch_size, k, n }, dt::f4_e2m1, { stride_b, n, 1 });
        const dnnl::memory::desc s_md_3d({ batch_size, groups, n }, dt::e8m0, { stride_scales, n, 1 });

        if (pp_woq_3d_enabled && tri_state[dev_id].load(std::memory_order_acquire) >= 0) {
            dnnl::primitive_attr attr3d;
            attr3d.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            const int          mask3d       = (1 << 0) | (1 << 1) | (1 << 2);
            dnnl::memory::dims group_dims3d = { 1, group_size, 1 };
            attr3d.set_scales(DNNL_ARG_WEIGHTS, mask3d, group_dims3d, dt::e8m0);
#    ifdef GGML_SYCL_F16
            attr3d.set_fpmath_mode(dnnl::fpmath_mode::f16, /* apply_to_int = */ true);
#    endif
            DnnlPrimitiveKey key3d{};
            key3d.m               = m;
            key3d.n               = n;
            key3d.k               = k;
            key3d.batches_a       = batch_size;
            key3d.batches_b       = batch_size;
            key3d.at              = at;
            key3d.bt              = dt::f4_e2m1;
            key3d.ct              = ct;
            key3d.stra0           = stride_a;
            key3d.stra1           = k;
            key3d.stra2           = 1;
            key3d.strb0           = stride_b;
            key3d.strb1           = n;
            key3d.strb2           = 1;
            key3d.strc0           = stride_c;
            key3d.strc1           = n;
            key3d.ldc             = n;
            key3d.batch_size      = batch_size;
            key3d.variant         = 3;  // woq_gemm_batch_mxfp4, 3-D grouped-batch scales
            key3d.woq_group_size  = group_size;
            key3d.woq_scales_mask = mask3d;
            key3d.woq_zp_mask     = 0;

            const DnnlCachedPrimitive * cached3d = cache.get_or_create(key3d, eng, a_md, b_md_3d, c_md, attr3d);
            if (cached3d) {
                tri_state[dev_id].store(1, std::memory_order_release);
                if (g_ggml_sycl_debug && arm_print_latch[dev_id].exchange(1, std::memory_order_relaxed) == 0) {
                    std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4-BATCH] device=%d arm=3D\n", dev_id);
                }
                auto b_mem = dnnl::memory(b_md_3d, eng, const_cast<void *>(b_nibbles));
                auto s_mem = dnnl::memory(s_md_3d, eng, const_cast<void *>(b_scales));

                std::unordered_map<int, dnnl::memory> args;
                args.insert({ DNNL_ARG_SRC, a_mem });
                args.insert({ DNNL_ARG_WEIGHTS, b_mem });
                args.insert({ DNNL_ARG_DST, c_mem });
                args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, s_mem });
                if (cached3d->scratchpad_md.get_size() > 0) {
                    auto scratchpad_mem = ctx.get_scratchpad_mem(cached3d->scratchpad_md, eng, q);
                    if (scratchpad_mem.get(true) == nullptr) {
                        throw std::runtime_error("oneDNN scratchpad allocation failed");
                    }
                    args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
                }
                // llama.cpp-iikr (sycl-kernel-profiler extension cycle, final
                // micro-cycle): switched from record-event to
                // ggml_sycl_profile_submit so the classifier gets a HOST
                // submit span, not just the device timestamps -- see the
                // 2-D fallback arm below for the full rationale (team-lead's
                // c-vr68 capture, no_submit_span). dnnl::sycl_interop::
                // execute() doesn't need the sycl::queue& submit_fn hands
                // it (it uses the dnnl::stream captured above instead), so
                // the lambda ignores that parameter and closes over
                // `stream`/`args`/`deps` -- *q is passed only to give
                // ggml_sycl_profile_submit a real queue to bracket with host
                // chrono.
                ggml_sycl_profile_label gemm_label{};
                gemm_label.name       = "mxfp4.pp.gemm.execute";
                gemm_label.category   = "mxfp4.pp.gemm";
                gemm_label.queue_kind = "compute";
                gemm_label.metadata   = "arm=3d";
                gemm_label.device     = ctx.device;
                return ggml_sycl_profile_submit(*q, gemm_label, [&](sycl::queue &) {
                    return dnnl::sycl_interop::execute(cached3d->primitive, stream, args, deps);
                });
            }
            tri_state[dev_id].store(-1, std::memory_order_release);
            if (g_ggml_sycl_debug) {
                std::fprintf(stderr,
                             "[ONEDNN][WOQ-MXFP4-BATCH] 3-D grouped-batch scales refused on device=%d; "
                             "falling back to per-batch 2-D primitives\n",
                             dev_id);
            }
        }

        // Fallback: C1's exact proven 2-D primitive (mask 3, group_dims
        // {group_size,1}), one primitive shared across the loop (pointer
        // offsets differ per batch element, shape does not), executed once
        // per batch element and merged into a single returned event.
        const dnnl::memory::desc a_md_2d({ m, k }, at, { k, 1 });
        const dnnl::memory::desc b_md_2d({ k, n }, dt::f4_e2m1, { n, 1 });
        const dnnl::memory::desc c_md_2d({ m, n }, ct, { n, 1 });
        const dnnl::memory::desc s_md_2d({ groups, n }, dt::e8m0, { n, 1 });

        const dnnl::primitive_attr attr2d = woq_mxfp4_2d_attr(group_size);
        DnnlPrimitiveKey           key2d{};
        key2d.m               = m;
        key2d.n               = n;
        key2d.k               = k;
        key2d.batches_a       = 1;
        key2d.batches_b       = 1;
        key2d.at              = at;
        key2d.bt              = dt::f4_e2m1;
        key2d.ct              = ct;
        // llama.cpp-sr83 (quality review finding #10): this primitive has no
        // batch dimension at all (batches_a/batches_b are 1; a_md_2d/b_md_2d
        // are plain 2-D descriptors), so there is no batch stride to record
        // here -- stra2/strb2/stride_a/stride_b/stride_c stay at their
        // zero-initialized default rather than holding a synthetic "total
        // size" value. m/n/k already differentiate distinct shapes in this
        // key; a non-stride number in a field named for a stride was noise,
        // not a correctness requirement.
        key2d.stra0           = 1;
        key2d.stra1           = k;
        key2d.strb0           = 1;
        key2d.strb1           = n;
        key2d.ldc             = n;
        key2d.variant         = 4;  // woq_gemm_batch_mxfp4, per-batch 2-D fallback
        key2d.woq_group_size  = group_size;
        key2d.woq_scales_mask = woq_mxfp4_2d_scale_mask;
        key2d.woq_zp_mask     = 0;

        const DnnlCachedPrimitive * cached2d = cache.get_or_create(key2d, eng, a_md_2d, b_md_2d, c_md_2d, attr2d);
        if (!cached2d) {
            throw std::runtime_error("woq_gemm_batch_mxfp4: 2-D fallback primitive creation failed");
        }
        if (g_ggml_sycl_debug && arm_print_latch[dev_id].exchange(1, std::memory_order_relaxed) == 0) {
            std::fprintf(stderr, "[ONEDNN][WOQ-MXFP4-BATCH] device=%d arm=2D-loop\n", dev_id);
        }

        // llama.cpp-sr83 (quality review finding #1): every iteration binds
        // DNNL_ARG_SCRATCHPAD to the SAME shared scratchpad buffer
        // (ctx.get_scratchpad_mem returns cached2d's one pooled allocation,
        // not a fresh one per call), so iteration b+1 reusing it before
        // iteration b's execute() has completed is a real hazard, not a
        // hypothetical one. Ordering = event edges, never submission order
        // (owner ruling, no-host-waits-event-chain-everything) -- an
        // in-order SYCL queue orders queue *submissions*, not oneDNN's
        // internal scratchpad access, so relying on submission order alone
        // is exactly the missing-edge pattern that ruling warns about. Chain
        // explicitly: iteration 0 waits on the caller's `deps`, iteration
        // b>0 waits on iteration b-1's own completion event. Free on an
        // in-order queue (the chain only makes explicit an order the queue
        // already provides for submission, while adding the correctness
        // guarantee for oneDNN's internal scratchpad reuse that submission
        // order alone does not).
        std::vector<sycl::event> per_batch_events;
        per_batch_events.reserve(static_cast<size_t>(batch_size));
        for (int b = 0; b < batch_size; ++b) {
            const char *    a_b  = static_cast<const char *>(a) + static_cast<int64_t>(b) * stride_a * elem_bytes;
            const uint8_t * bn_b = static_cast<const uint8_t *>(b_nibbles) + (static_cast<int64_t>(b) * stride_b) / 2;
            const uint8_t * bs_b = static_cast<const uint8_t *>(b_scales) + static_cast<int64_t>(b) * stride_scales;
            char *          c_b  = static_cast<char *>(c) + static_cast<int64_t>(b) * stride_c * out_elem_bytes;

            auto a_mem_b = dnnl::memory(a_md_2d, eng, const_cast<char *>(a_b));
            auto b_mem_b = dnnl::memory(b_md_2d, eng, const_cast<uint8_t *>(bn_b));
            auto s_mem_b = dnnl::memory(s_md_2d, eng, const_cast<uint8_t *>(bs_b));
            auto c_mem_b = dnnl::memory(c_md_2d, eng, c_b);

            // llama.cpp-iikr (unphased-host-time cycle): bracket the fresh
            // per-iteration map construction team-lead named as a bisection
            // candidate -- see ggml_sycl_gemm_profile's declaration comment
            // above for why this is unconditional and drained by exchange(0)
            // rather than gated on ggml-sycl.cpp's profiling flag.
            const auto                            args_map_t0 = std::chrono::high_resolution_clock::now();
            std::unordered_map<int, dnnl::memory> args;
            args.insert({ DNNL_ARG_SRC, a_mem_b });
            args.insert({ DNNL_ARG_WEIGHTS, b_mem_b });
            args.insert({ DNNL_ARG_DST, c_mem_b });
            args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, s_mem_b });
            if (cached2d->scratchpad_md.get_size() > 0) {
                auto scratchpad_mem = ctx.get_scratchpad_mem(cached2d->scratchpad_md, eng, q);
                if (scratchpad_mem.get(true) == nullptr) {
                    throw std::runtime_error("oneDNN scratchpad allocation failed");
                }
                args.insert({ DNNL_ARG_SCRATCHPAD, scratchpad_mem });
            }
            {
                const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::high_resolution_clock::now() - args_map_t0)
                                       .count();
                ggml_sycl_gemm_profile::args_map_ns_accum().fetch_add(ns, std::memory_order_relaxed);
                ggml_sycl_gemm_profile::args_map_calls_accum().fetch_add(1, std::memory_order_relaxed);
            }
            const std::vector<sycl::event> iter_deps =
                (b == 0) ? deps : std::vector<sycl::event>{ per_batch_events.back() };
            // llama.cpp-iikr (sycl-kernel-profiler extension cycle): THIS is
            // the required arm -- the per-expert-execute loop the static
            // survey (task comment c-d92w) found completely dark to the
            // gap-cause classifier, and the one team-lead's three suspects
            // (queue occupancy of many small submissions, cross-engine
            // dependency latency, host graph-walk overhead) most needed
            // resolved. One event per batch element, matching the loop's
            // own per-expert granularity -- deliberately NOT the merged
            // barrier returned below, which times only the (near-instant)
            // barrier command itself (see this function's own gemm_events
            // caller-side comment in ggml-sycl.cpp for that exact trap).
            // Final micro-cycle: switched to ggml_sycl_profile_submit (host
            // submit-span recording, not just record-event's device-only
            // timestamps) -- team-lead's full-coverage capture (c-vr68)
            // found 890/934 ms landed in the classifier's no_submit_span
            // instrument-artifact cause specifically because the record-
            // event form this arm used before captured no host submit span,
            // so the parser could not tell host-late from submitted-early-
            // and-queued. batch_metadata is a local std::string kept alive
            // for the whole ggml_sycl_profile_submit call (a single full
            // expression) -- ggml_sycl_kernel_profile_record_event, called
            // from inside it, synchronously copies .c_str() into an owned
            // std::string (profile_key::metadata) before returning, so this
            // remains safe -- see snapshot_label in sycl-kernel-profiler.cpp.
            // The lambda ignores the sycl::queue& submit_fn hands it for the
            // same reason as the 3-D arm above (dnnl::sycl_interop::execute
            // uses the captured dnnl::stream, not a raw sycl::queue).
            ggml_sycl_profile_label gemm_label{};
            gemm_label.name                  = "mxfp4.pp.gemm.execute";
            gemm_label.category              = "mxfp4.pp.gemm";
            gemm_label.queue_kind            = "compute";
            const std::string batch_metadata = "arm=2d;batch_idx=" + std::to_string(b);
            gemm_label.metadata              = batch_metadata.c_str();
            gemm_label.device                = ctx.device;
            sycl::event exec_event           = ggml_sycl_profile_submit(*q, gemm_label, [&](sycl::queue &) {
                return dnnl::sycl_interop::execute(cached2d->primitive, stream, args, iter_deps);
            });
            per_batch_events.push_back(exec_event);
        }
        return q->ext_oneapi_submit_barrier(per_batch_events);
    }

    // Pointer array batch GEMM - C[i] = alpha * A[i] * B[i] + beta * C[i]
    // For arrays of matrix pointers (non-contiguous batches)
    // Falls back to iterating over individual GEMM operations
    static void gemm_batch_array(ggml_backend_sycl_context & ctx,
                                 bool                        trans_a,
                                 bool                        trans_b,
                                 int                         m,
                                 int                         n,
                                 int                         k,
                                 float                       alpha,
                                 const void **               a,
                                 dt                          at,
                                 int                         lda,
                                 const void **               b,
                                 dt                          bt,
                                 int                         ldb,
                                 float                       beta,
                                 void **                     c,
                                 dt                          ct,
                                 int                         ldc,
                                 int                         batch_size,
                                 const queue_ptr &           q) {
        // For pointer arrays, we iterate and call individual GEMM operations
        // This is less efficient than strided batch but handles non-contiguous data
        for (int i = 0; i < batch_size; ++i) {
            gemm_batch_strided(ctx, trans_a, trans_b, m, n, k, alpha, a[i], at, lda, 0, b[i], bt, ldb, 0, beta, c[i],
                               ct, ldc, 0, 1, q);
        }
    }

    // Simplified row-major batch GEMM (no transpose, alpha=1, beta=0)
    static void row_gemm_batch(ggml_backend_sycl_context & ctx,
                               int                         m,
                               int                         n,
                               int                         k,
                               const void *                a,
                               dt                          at,
                               int64_t                     stride_a,
                               const void *                b,
                               dt                          bt,
                               int64_t                     stride_b,
                               void *                      c,
                               dt                          ct,
                               [[maybe_unused]] int64_t    stride_c,
                               int                         batch_size,
                               const queue_ptr &           q) {
        // Use the existing gemm function which handles batching natively
        gemm(ctx, m, n, k, a, at, 1, k, stride_a, b, bt, 1, k, stride_b, c, ct, q, batch_size, batch_size);
    }
};

#endif

#endif  // GGML_SYCL_GEMM_HPP
