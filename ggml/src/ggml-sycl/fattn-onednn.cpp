//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn-onednn.hpp"

#if GGML_SYCL_DNNL

#    include "common.hpp"
#    include "fattn-common.hpp"

#    include <cmath>
#    include <cstdio>
#    include <mutex>

using namespace dnnl::graph;
using lt       = logical_tensor;
using layout_t = lt::layout_type;
using dim_t    = lt::dim;
using dims_t   = lt::dims;
using dt       = lt::data_type;

// Per-context SDPA cache. Stored as an opaque unique_ptr inside
// ggml_backend_sycl_context to avoid pulling graph headers into common.hpp.
// Created on first use, freed at context teardown via SdpaCacheDeleter.
//
// This global mutex guards ONLY the cache allocation race (first call on a given
// context). Once the cache exists, all further accesses use the per-cache
// `cache->m` — so multi-GPU workloads with one cache per context do not serialise
// across devices.
static std::mutex g_sdpa_cache_init_mutex;

static sdpa_partition_cache * get_or_create_cache(ggml_backend_sycl_context & ctx) {
    std::lock_guard<std::mutex> lock(g_sdpa_cache_init_mutex);
    if (!ctx.sdpa_cache) {
        auto cache = std::make_unique<sdpa_partition_cache>();
        ctx.sdpa_cache.reset(cache.release());
    }
    return static_cast<sdpa_partition_cache *>(ctx.sdpa_cache.get());
}

void ggml_sycl_sdpa_cache_destroy(void * ptr) {
    std::unique_ptr<sdpa_partition_cache> cache(static_cast<sdpa_partition_cache *>(ptr));
    // Two separate safety claims at play here:
    //   1. The CACHE (this object) is never destroyed concurrently with an
    //      inflight FA dispatch — ggml's backend-teardown contract runs
    //      after all pending graph work has drained. That is what makes
    //      destroying `cache` from here safe without a lock on `cache->m`.
    //   2. Entries in `cache->hits` (shared_ptr<sdpa_compiled_entry>) are
    //      separately shielded: any dispatch call that observed an entry
    //      holds its own shared_ptr across the unlock, so the entry
    //      outlives the map even if the map or cache were mutated. That
    //      property is used by the dispatch entry, NOT by this destroy —
    //      it is not what makes `cache` destruction below legal.
    // Per-entry USM scratch (`scale_owner`) is released by sdpa_compiled_entry's
    // mem_handle when the last shared_ptr drops during `cache` destruction.
}

// -------------------------------------------------------------------
// Eligibility gate
// -------------------------------------------------------------------
static ggml_sycl_onednn_fa_layout_plan ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason reason) {
    return { ggml_sycl_onednn_fa_layout_kind::REJECT, reason };
}

class onednn_fa_materialize_k_kernel;
class onednn_fa_materialize_v_kernel;
class onednn_fa_materialize_q_f32_kernel;
class onednn_fa_materialized_release_marker_kernel;

ggml_sycl_onednn_fa_layout_plan ggml_sycl_flash_attn_ext_onednn_plan(const fattn_params & params,
                                                                     int                  H_q,
                                                                     int                  H_kv,
                                                                     bool                 kv_is_fp8,
                                                                     bool                 multi_seq) {
    // Default threshold is 8 to avoid excessive JIT recompilation during TG.
    // GGML_SYCL_FA_ONEDNN_MIN_NCOLS=0 disables the threshold (all batch sizes).
    static const int onednn_min_ncols = []() {
        const char * e = std::getenv("GGML_SYCL_FA_ONEDNN_MIN_NCOLS");
        return e ? std::atoi(e) : 8;
    }();
    if (params.ne01 < onednn_min_ncols) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::BELOW_MIN_NCOLS);
    }
    if (params.sinks) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::SINKS_UNSUPPORTED);
    }
    if (params.logit_softcap != 0) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::SOFTCAP_UNSUPPORTED);
    }
    if (params.max_bias != 0.0f) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::MAX_BIAS_UNSUPPORTED);
    }
    if (kv_is_fp8) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::FP8_KV_UNSUPPORTED);
    }
    if (multi_seq) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::MULTI_SEQ_UNSUPPORTED);
    }
    if (params.ne03 != 1 || params.ne13 != 1 || (params.mask && params.ne33 != 1)) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::BATCH_UNSUPPORTED);
    }
    if (params.use_paged_attn || params.use_paged_layout || params.block_table || params.seq_lens) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::PAGED_UNSUPPORTED);
    }
    if (params.ne00 > 512) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D);
    }
    if (params.ne11 <= 0) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::EMPTY_KV);
    }
    if (H_q <= 0 || H_kv <= 0 || (H_q % H_kv) != 0) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::HEAD_RATIO_UNSUPPORTED);
    }
    // The compiled partition is built with Q/K/V dtypes taken from
    // params.*_type. The scalar divisor is allocated with the same dtype as Q,
    // so Mistral-style F32 Q + F16 KV can use the oneDNN PP path.
    if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::Q_TYPE_UNSUPPORTED);
    }
    if (params.K_type != GGML_TYPE_F16) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::K_TYPE_UNSUPPORTED);
    }
    if (params.V_type != GGML_TYPE_F16) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::V_TYPE_UNSUPPORTED);
    }
    if (params.mask && params.mask_type != GGML_TYPE_F16 && params.mask_type != GGML_TYPE_F32) {
        return ggml_sycl_onednn_fa_reject(ggml_sycl_onednn_fa_layout_reason::MASK_TYPE_UNSUPPORTED);
    }

    // oneDNN Graph SDPA can represent MHA as 4-D (batch,H,S,D) and GQA/MQA as
    // 5-D (batch,H_kv,N_rep,S,D). The direct GQA/MQA path still requires a
    // contiguous K/V token-D plane. When nc_stride != D, the planner asks the
    // unified-cache-backed materializer for dense f16 K/V before oneDNN execute.
    const int64_t k_nc_stride = params.nb11 / (int64_t) sizeof(sycl::half);
    const int64_t v_nc_stride = params.nb21 / (int64_t) sizeof(sycl::half);
    const int64_t D           = params.ne00;
    if ((H_q != H_kv) && (k_nc_stride != D || v_nc_stride != D)) {
        return { ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED,
                 ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH };
    }
    (void) H_q;
    (void) H_kv;
    return { ggml_sycl_onednn_fa_layout_kind::DIRECT, ggml_sycl_onednn_fa_layout_reason::OK };
}

bool ggml_sycl_flash_attn_ext_onednn_eligible(const fattn_params & params,
                                              int                  H_q,
                                              int                  H_kv,
                                              bool                 kv_is_fp8,
                                              bool                 multi_seq) {
    const ggml_sycl_onednn_fa_layout_plan plan =
        ggml_sycl_flash_attn_ext_onednn_plan(params, H_q, H_kv, kv_is_fp8, multi_seq);
    return plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT;
}

bool ggml_sycl_flash_attn_ext_onednn_materialization_desc(const fattn_params &                       params,
                                                          int                                        H_q,
                                                          int                                        H_kv,
                                                          int                                        target_device,
                                                          ggml_sycl_onednn_fa_materialization_desc * out) {
    if (!out) {
        return false;
    }
    *out = {};

    const ggml_sycl_onednn_fa_layout_plan plan =
        ggml_sycl_flash_attn_ext_onednn_plan(params, H_q, H_kv, params.kv_is_fp8, params.n_seqs > 1);
    if (plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT) {
        return false;
    }
    if (H_q <= 0 || H_kv <= 0 || (H_q % H_kv) != 0) {
        return false;
    }

    const int64_t elem_size = (int64_t) sizeof(sycl::half);
    if (params.ne00 <= 0 || params.ne11 <= 0 || params.ne12 <= 0 || params.ne10 != params.ne00) {
        return false;
    }
    if (params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
        return false;
    }
    if (params.nb11 < elem_size || params.nb12 < elem_size || params.nb21 < elem_size || params.nb22 < elem_size) {
        return false;
    }

    const int64_t dense_token_nb = (int64_t) params.ne00 * elem_size;
    const int64_t dense_head_nb  = dense_token_nb * (int64_t) params.ne11;
    const int64_t dense_batch_nb = dense_head_nb * (int64_t) params.ne12;

    out->required         = (plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED);
    out->target_device    = target_device;
    out->D                = params.ne00;
    out->n_kv             = params.ne11;
    out->H_q              = H_q;
    out->H_kv             = H_kv;
    out->n_rep            = H_q / H_kv;
    out->bytes_per_tensor = out->required ? (size_t) dense_batch_nb : 0;
    out->k_src_nb1        = params.nb11;
    out->k_src_nb2        = params.nb12;
    out->k_src_nb3        = params.nb13;
    out->v_src_nb1        = params.nb21;
    out->v_src_nb2        = params.nb22;
    out->v_src_nb3        = params.nb23;
    out->k_target_nb1     = dense_token_nb;
    out->k_target_nb2     = dense_head_nb;
    out->k_target_nb3     = dense_batch_nb;
    out->v_target_nb1     = dense_token_nb;
    out->v_target_nb2     = dense_head_nb;
    out->v_target_nb3     = dense_batch_nb;
    return true;
}

size_t ggml_sycl_onednn_fa_materialized_src_offset(const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                   bool                                             value_tensor,
                                                   int                                              h,
                                                   int                                              t,
                                                   int                                              d) {
    const int64_t nb1 = value_tensor ? desc.v_src_nb1 : desc.k_src_nb1;
    const int64_t nb2 = value_tensor ? desc.v_src_nb2 : desc.k_src_nb2;
    return (size_t) h * (size_t) nb2 + (size_t) t * (size_t) nb1 + (size_t) d * sizeof(sycl::half);
}

size_t ggml_sycl_onednn_fa_materialized_dst_offset(const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                   int                                              h,
                                                   int                                              t,
                                                   int                                              d) {
    return (size_t) h * (size_t) desc.k_target_nb2 + (size_t) t * (size_t) desc.k_target_nb1 +
           (size_t) d * sizeof(sycl::half);
}

static sycl::event ggml_sycl_onednn_fa_materialize_one(sycl::queue &                                    stream,
                                                       const char *                                     src,
                                                       char *                                           dst,
                                                       const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                       bool                                             value_tensor) {
    const int H_kv = desc.H_kv;
    const int n_kv = desc.n_kv;
    const int D    = desc.D;

    const int64_t src_nb1 = value_tensor ? desc.v_src_nb1 : desc.k_src_nb1;
    const int64_t src_nb2 = value_tensor ? desc.v_src_nb2 : desc.k_src_nb2;
    const int64_t dst_nb1 = value_tensor ? desc.v_target_nb1 : desc.k_target_nb1;
    const int64_t dst_nb2 = value_tensor ? desc.v_target_nb2 : desc.k_target_nb2;

    if (value_tensor) {
        return stream.parallel_for<onednn_fa_materialize_v_kernel>(
            sycl::range<3>((size_t) H_kv, (size_t) n_kv, (size_t) D), [=](sycl::id<3> idx) {
                const int    h = (int) idx[0];
                const int    t = (int) idx[1];
                const int    d = (int) idx[2];
                const size_t src_off =
                    (size_t) h * (size_t) src_nb2 + (size_t) t * (size_t) src_nb1 + (size_t) d * sizeof(sycl::half);
                const size_t dst_off =
                    (size_t) h * (size_t) dst_nb2 + (size_t) t * (size_t) dst_nb1 + (size_t) d * sizeof(sycl::half);
                reinterpret_cast<sycl::half *>(dst + dst_off)[0] =
                    reinterpret_cast<const sycl::half *>(src + src_off)[0];
            });
    }
    return stream.parallel_for<onednn_fa_materialize_k_kernel>(
        sycl::range<3>((size_t) H_kv, (size_t) n_kv, (size_t) D), [=](sycl::id<3> idx) {
            const int    h = (int) idx[0];
            const int    t = (int) idx[1];
            const int    d = (int) idx[2];
            const size_t src_off =
                (size_t) h * (size_t) src_nb2 + (size_t) t * (size_t) src_nb1 + (size_t) d * sizeof(sycl::half);
            const size_t dst_off =
                (size_t) h * (size_t) dst_nb2 + (size_t) t * (size_t) dst_nb1 + (size_t) d * sizeof(sycl::half);
            reinterpret_cast<sycl::half *>(dst + dst_off)[0] = reinterpret_cast<const sycl::half *>(src + src_off)[0];
        });
}

static bool ggml_sycl_flash_attn_ext_onednn_materialize_q_f32(const fattn_params &                  params,
                                                              int                                   target_device,
                                                              sycl::queue &                         stream,
                                                              ggml_sycl_onednn_fa_materialized_kv * out,
                                                              int32_t *                             out_nb1,
                                                              int64_t *                             out_nb2,
                                                              int64_t *                             out_nb3) {
    if (!out || !out_nb1 || !out_nb2 || !out_nb3 || params.Q_type != GGML_TYPE_F32 || !params.Q) {
        return false;
    }
    if (params.ne00 <= 0 || params.ne01 <= 0 || params.ne02 <= 0 || params.ne03 != 1 || target_device < 0) {
        return false;
    }

    const int64_t D       = params.ne00;
    const int64_t n_q     = params.ne01;
    const int64_t H_q     = params.ne02;
    const int64_t dst_nb1 = D * (int64_t) sizeof(sycl::half);
    const int64_t dst_nb2 = dst_nb1 * n_q;
    const int64_t dst_nb3 = dst_nb2 * H_q;
    const size_t  q_bytes = (size_t) dst_nb3;

    ggml_sycl::alloc_request req{};
    req.queue                               = &stream;
    req.device                              = target_device;
    req.size                                = q_bytes;
    req.intent.role                         = ggml_sycl::alloc_role::GRAPH_TMP;
    req.intent.category                     = ggml_sycl::runtime_category::GRAPH;
    req.intent.cohort_id                    = "onednn-fa-q-materialized";
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;

    out->Q = ggml_sycl::unified_allocate(req);
    if (!out->Q.valid()) {
        return false;
    }
    auto q_resolved = out->Q.resolve(target_device);
    if (!q_resolved || !q_resolved.on_device) {
        out->Q = {};
        return false;
    }

    char *        dst     = static_cast<char *>(q_resolved.ptr);
    const char *  src     = params.Q;
    const int64_t src_nb1 = params.nb01;
    const int64_t src_nb2 = params.nb02;

    try {
        stream
            .parallel_for<onednn_fa_materialize_q_f32_kernel>(
                sycl::range<3>((size_t) H_q, (size_t) n_q, (size_t) D),
                [=](sycl::id<3> idx) {
                    const int64_t h = (int64_t) idx[0];
                    const int64_t t = (int64_t) idx[1];
                    const int64_t d = (int64_t) idx[2];
                    const size_t  src_off =
                        (size_t) h * (size_t) src_nb2 + (size_t) t * (size_t) src_nb1 + (size_t) d * sizeof(float);
                    const size_t dst_off =
                        (size_t) h * (size_t) dst_nb2 + (size_t) t * (size_t) dst_nb1 + (size_t) d * sizeof(sycl::half);
                    reinterpret_cast<sycl::half *>(dst + dst_off)[0] =
                        static_cast<sycl::half>(reinterpret_cast<const float *>(src + src_off)[0]);
                })
            .wait_and_throw();
    } catch (const std::exception & e) {
        if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            fprintf(stderr, "[SYCL] fattn: oneDNN materialized Q repack failed: %s\n", e.what());
        }
        out->Q = {};
        return false;
    } catch (...) {
        if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            fprintf(stderr, "[SYCL] fattn: oneDNN materialized Q repack failed with unknown exception\n");
        }
        out->Q = {};
        return false;
    }

    *out_nb1 = (int32_t) dst_nb1;
    *out_nb2 = dst_nb2;
    *out_nb3 = dst_nb3;
    return true;
}

bool ggml_sycl_flash_attn_ext_onednn_materialize_kv(const ggml_sycl_onednn_fa_materialization_desc & desc,
                                                    const fattn_params &                             params,
                                                    sycl::queue &                                    stream,
                                                    ggml_sycl_onednn_fa_materialized_kv *            out) {
    if (!out) {
        return false;
    }
    *out = {};
    if (!desc.required) {
        return true;
    }
    if (!params.K || !params.V || desc.bytes_per_tensor == 0 || desc.target_device < 0) {
        return false;
    }

    ggml_sycl::alloc_request req{};
    req.queue                               = &stream;
    req.device                              = desc.target_device;
    req.size                                = desc.bytes_per_tensor;
    req.intent.role                         = ggml_sycl::alloc_role::KV;
    req.intent.category                     = ggml_sycl::runtime_category::KV_CACHE;
    req.intent.cohort_id                    = "onednn-fa-kv-materialized";
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;

    out->K = ggml_sycl::unified_allocate(req);
    if (!out->K.valid()) {
        return false;
    }
    out->V = ggml_sycl::unified_allocate(req);
    if (!out->V.valid()) {
        out->K = {};
        return false;
    }

    auto k_resolved = out->K.resolve(desc.target_device);
    auto v_resolved = out->V.resolve(desc.target_device);
    if (!k_resolved || !v_resolved || !k_resolved.on_device || !v_resolved.on_device) {
        out->K = {};
        out->V = {};
        return false;
    }

    try {
        sycl::event k_evt = ggml_sycl_onednn_fa_materialize_one(stream, params.K, static_cast<char *>(k_resolved.ptr),
                                                                desc, /*value_tensor=*/false);
        sycl::event v_evt = ggml_sycl_onednn_fa_materialize_one(stream, params.V, static_cast<char *>(v_resolved.ptr),
                                                                desc, /*value_tensor=*/true);
        v_evt.wait_and_throw();
        k_evt.wait_and_throw();
    } catch (const std::exception & e) {
        if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            fprintf(stderr, "[SYCL] fattn: oneDNN materialized K/V repack failed: %s\n", e.what());
        }
        out->K = {};
        out->V = {};
        return false;
    } catch (...) {
        if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            fprintf(stderr, "[SYCL] fattn: oneDNN materialized K/V repack failed with unknown exception\n");
        }
        out->K = {};
        out->V = {};
        return false;
    }
    return true;
}

// -------------------------------------------------------------------
// Graph builder helpers
// -------------------------------------------------------------------

// Build logical tensors and ops for the 5-op SDPA pattern.
// Returns the list of input/output logical_tensors needed for compile and execute.
//
// Pattern: MatMul(Q, K^T) -> Divide(scale) -> [Add(mask)] -> SoftMax -> MatMul(V)
//
// For non-GQA (H_q == H_kv): Q/K/V are 4-D (batch, H, S, D).
// For GQA     (H_q != H_kv): Q is 5-D (batch, H_kv, N_rep, S, D);
//                              K/V are 5-D (batch, H_kv, 1, S, D).
// The same pointer is used — only the logical_tensor dims/strides differ.
// RAII wrapper for build output. Owns `scale_usm` through `scale_owner` until
// the caller transfers it into `sdpa_compiled_entry`. If any exception fires
// between build_and_compile_sdpa returning and the transfer completing (e.g.
// std::make_shared bad_alloc), the mem_handle destructor releases the buffer
// instead of leaking.
struct build_result {
    std::vector<lt>                 in_ports;
    std::vector<lt>                 out_ports;
    dnnl::graph::compiled_partition cp;
    ggml_sycl::mem_handle           scale_owner;
    void *                          scale_usm = nullptr;

    build_result()                                 = default;
    build_result(const build_result &)             = delete;
    build_result & operator=(const build_result &) = delete;

    build_result(build_result && o) noexcept :
        in_ports(std::move(o.in_ports)),
        out_ports(std::move(o.out_ports)),
        cp(std::move(o.cp)),
        scale_owner(std::move(o.scale_owner)),
        scale_usm(o.scale_usm) {
        o.scale_usm = nullptr;
    }

    build_result & operator=(build_result && o) noexcept {
        if (this != &o) {
            in_ports    = std::move(o.in_ports);
            out_ports   = std::move(o.out_ports);
            cp          = std::move(o.cp);
            scale_owner = std::move(o.scale_owner);
            scale_usm   = o.scale_usm;
            o.scale_usm = nullptr;
        }
        return *this;
    }

    ~build_result() = default;
};

static build_result build_and_compile_sdpa(const sdpa_shape_key & key, const dnnl::engine & eng, sycl::queue * stream) {
    const int  batch  = 1;
    const int  D      = key.D;
    const int  ncols  = key.ncols;
    const int  ne11   = key.ne11;
    const int  H_q    = key.H_q;
    const int  H_kv   = key.H_kv;
    const int  N_rep  = (H_kv > 0) ? (H_q / H_kv) : 1;
    const bool is_gqa = key.is_gqa;

    // Element-size lookup for the source types stored in the key.
    const size_t q_esz = ggml_type_size(static_cast<ggml_type>(key.Q_type));
    const size_t k_esz = ggml_type_size(static_cast<ggml_type>(key.K_type));
    const size_t v_esz = ggml_type_size(static_cast<ggml_type>(key.V_type));
    const size_t m_esz = key.has_mask ? ggml_type_size(static_cast<ggml_type>(key.mask_type)) : 1;

    // Map ggml_type -> oneDNN data_type for sources.
    auto to_dnnl_dt = [](int ggml_t) -> dt {
        switch (static_cast<ggml_type>(ggml_t)) {
            case GGML_TYPE_F16:
                return dt::f16;
            case GGML_TYPE_F32:
                return dt::f32;
            default:
                throw std::runtime_error("oneDNN SDPA: unsupported ggml_type for source tensor");
        }
    };

    const dt q_dt    = to_dnnl_dt(key.Q_type);
    const dt k_dt    = to_dnnl_dt(key.K_type);
    const dt v_dt    = to_dnnl_dt(key.V_type);
    const dt mask_dt = key.has_mask ? to_dnnl_dt(key.mask_type) : dt::f32;

    // Logical tensor IDs must be unique across the entire graph.
    size_t id = 0;

    // ---- Q / K / V shapes -------------------------------------------------
    // Strides for the source tensors come from the ggml tensor byte strides
    // (key.*_nb*) divided by element size. This is correct under arbitrary
    // ggml layouts including non-contiguous / permuted sources where the
    // hardcoded contiguous-stride formula would read garbage.
    //
    // ggml byte strides: nb[0]=elem_size, nb[1]=row stride, nb[2]=head stride,
    //                    nb[3]=batch stride. We use nb[1..3]; the innermost
    //                    element stride is always 1 element.

    dims_t q_dims, k_dims, v_dims, score_dims, out_dims;
    dims_t q_strides, k_strides, v_strides, score_strides, out_strides;

    // Element strides from ggml tensor nb (bytes) / element size.
    const dim_t q_s1 = static_cast<dim_t>(key.q_nb1 / (int64_t) q_esz);  // stride along Q rows (ncols dim)
    const dim_t q_s2 = static_cast<dim_t>(key.q_nb2 / (int64_t) q_esz);  // stride along Q heads
    const dim_t q_s3 = static_cast<dim_t>(key.q_nb3 / (int64_t) q_esz);  // stride along Q batch

    const dim_t k_s1 = static_cast<dim_t>(key.k_nb1 / (int64_t) k_esz);
    const dim_t k_s2 = static_cast<dim_t>(key.k_nb2 / (int64_t) k_esz);
    const dim_t k_s3 = static_cast<dim_t>(key.k_nb3 / (int64_t) k_esz);

    const dim_t v_s1 = static_cast<dim_t>(key.v_nb1 / (int64_t) v_esz);
    const dim_t v_s2 = static_cast<dim_t>(key.v_nb2 / (int64_t) v_esz);
    const dim_t v_s3 = static_cast<dim_t>(key.v_nb3 / (int64_t) v_esz);

    // Output is a dense f32 buffer in params.dst laid out as (batch, H_q, ncols, D).
    if (!is_gqa) {
        // 4-D non-GQA: (batch, H, S, D)
        q_dims     = { batch, H_q, ncols, D };
        k_dims     = { batch, H_kv, ne11, D };
        v_dims     = { batch, H_kv, ne11, D };
        score_dims = { batch, H_q, ncols, ne11 };
        out_dims   = { batch, H_q, ncols, D };

        // Q logical dims are (batch, H, ncols, D) mapped from ggml Q[D, ncols, H, batch]
        // dim0=batch <- nb3, dim1=H <- nb2, dim2=ncols <- nb1, dim3=D <- elem stride 1
        q_strides = { q_s3, q_s2, q_s1, 1 };
        k_strides = { k_s3, k_s2, k_s1, 1 };
        v_strides = { v_s3, v_s2, v_s1, 1 };

        // Score is an internal contiguous intermediate. Output writes directly
        // to ggml dst layout [D, H, ncols, batch], viewed as oneDNN logical
        // dims (batch, H, ncols, D).
        score_strides = { H_q * ncols * ne11, ncols * ne11, ne11, 1 };
        out_strides   = { H_q * ncols * D, D, H_q * D, 1 };
    } else {
        // 5-D GQA: Q=(batch, H_kv, N_rep, ncols, D), K/V=(batch, H_kv, 1, ne11, D)
        q_dims     = { batch, H_kv, N_rep, ncols, D };
        k_dims     = { batch, H_kv, 1, ne11, D };
        v_dims     = { batch, H_kv, 1, ne11, D };
        score_dims = { batch, H_kv, N_rep, ncols, ne11 };
        out_dims   = { batch, H_kv, N_rep, ncols, D };

        // Q is laid out in ggml as (D, ncols, H_q, batch) with H_q = H_kv * N_rep.
        // We reinterpret the H_q axis as (H_kv, N_rep) with head index
        //   h_q = kv_head * N_rep + rep
        // Stride along kv_head = N_rep * (stride along h_q) = N_rep * q_s2
        // Stride along rep     = q_s2
        q_strides = { q_s3, static_cast<dim_t>((int64_t) N_rep * q_s2), q_s2, q_s1, 1 };

        // K/V: ggml layout (D, ne11, H_kv, batch). The N_rep axis in the 5-D
        // view is broadcast — we express it with a zero-ish stride by reusing
        // the kv_head stride, since all N_rep reps share the same KV slice.
        k_strides = { k_s3, k_s2,
                      k_s2,  // N_rep axis == 1 → same stride as kv_head (broadcast)
                      k_s1, 1 };
        v_strides = { v_s3, v_s2, v_s2, v_s1, 1 };

        score_strides = { (dim_t) (H_kv * N_rep * ncols * ne11), (dim_t) (N_rep * ncols * ne11), (dim_t) (ncols * ne11),
                          (dim_t) (ne11), (dim_t) (1) };
        // Dense f32 ggml output layout [D, H_q, ncols, batch], viewed 5-D as
        // (batch, H_kv, N_rep, ncols, D) with h_q = kv_head*N_rep + rep.
        out_strides   = { (dim_t) (H_kv * N_rep * ncols * D), (dim_t) (N_rep * D), (dim_t) D, (dim_t) (H_q * D),
                          (dim_t) 1 };
    }

    // ---- Scale tensor (scalar, same dtype as Q) ---------------------------
    const dims_t scale_dims    = { 1 };
    const dims_t scale_strides = { 1 };

    // ---- Mask tensor -------------------------------------------------------
    // ggml mask shape: (ne30=ne11, ne31=ncols_padded, ne32=H_or_1, ne33=batch).
    // The row (ncols) dimension is padded (GGML_KQ_MASK_PAD) so the physical
    // row stride nb31 is != ncols*elem_size — use the real tensor byte stride.
    //
    // We only derive m_s1 (element stride along the ncols axis). A per-head
    // stride m_s2 was intentionally dropped: the oneDNN logical_tensor dims
    // we emit have extent 1 on every head axis (ggml carries a non-per-head
    // mask with ne32=1, which is the only shape this dispatch path serves),
    // and the canonical oneDNN expression for a broadcast/extent-1 dim is
    // stride = 0 (see /opt/intel/oneapi/dnnl/2025.3/share/doc/dnnl/examples/
    // {gqa,sdpa}.cpp which use layout_type::strided + the natural dense
    // extent-1 stride). oneDNN never multiplies an extent-1 dim by its
    // stride when computing offsets, so a non-zero stride in those slots
    // reads the same memory; we still prefer 0 for intent clarity.
    //
    // Verified on Mistral-7B Q4_0 on Arc B580 (level_zero:0, seed 42 temp 0):
    // stride=0 in the head-axis slots produces the canonical
    // `6, 7, 8, 9, 10` smoke output.
    const dim_t m_s1 = key.has_mask ? static_cast<dim_t>(key.m_nb1 / (int64_t) m_esz) : 1;

    dims_t mask_dims, mask_strides;
    if (!is_gqa) {
        mask_dims    = { batch, 1, ncols, ne11 };
        mask_strides = { 0, 0, m_s1, 1 };
    } else {
        // 5-D GQA view: both head axes (H_kv, N_rep) have extent 1.
        mask_dims    = { batch, 1, 1, ncols, ne11 };
        mask_strides = { 0, 0, 0, m_s1, 1 };
    }

    // ---- Build logical tensors --------------------------------------------
    lt lt_q(id++, q_dt, q_dims, q_strides);
    lt lt_k(id++, k_dt, k_dims, k_strides);
    lt lt_scale(id++, q_dt, scale_dims, scale_strides);
    lt lt_v(id++, v_dt, v_dims, v_strides);

    // Intermediate f32 tensors
    lt lt_score(id++, dt::f32, score_dims, score_strides);
    lt lt_scaled(id++, dt::f32, score_dims, score_strides);
    lt lt_masked(id++, dt::f32, score_dims, score_strides);
    lt lt_probs(id++, q_dt, score_dims, score_strides);
    // Output is f32 directly into params.dst (no intermediate f16 buffer + cast).
    lt lt_out(id++, dt::f32, out_dims, out_strides);

    // ---- Ops ---------------------------------------------------------------
    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({ lt_q, lt_k });
    bmm1.add_outputs({ lt_score });

    // oneDNN's SDPA pattern matcher is canonical — it matches MatMul → Divide,
    // NOT MatMul → Multiply (even though they are mathematically equivalent with
    // the reciprocal scalar). Using Multiply here splits the fused SDPA partition
    // and pushes execution onto a slower, less-tested path that has been observed
    // to produce garbled output on Mistral 7B. See
    // /opt/intel/oneapi/dnnl/.../examples/{gqa,sdpa}.cpp for the reference
    // canonical pattern. The scalar written at execute time compensates by
    // storing sqrt(D) (i.e. 1/params.scale) instead of params.scale.
    op div_op(id++, op::kind::Divide, "scale_div");
    div_op.add_inputs({ lt_score, lt_scale });
    div_op.add_outputs({ lt_scaled });

    // mask_add and softmax inputs depend on whether mask is present
    lt              lt_sfmx_in = lt_scaled;
    std::vector<lt> mask_lts;
    op *            add_op_ptr = nullptr;
    op              add_op_storage(id++, op::kind::Add, "mask_add");

    if (key.has_mask) {
        lt lt_mask(id++, mask_dt, mask_dims, mask_strides);
        add_op_storage.add_inputs({ lt_scaled, lt_mask });
        add_op_storage.add_outputs({ lt_masked });
        add_op_ptr = &add_op_storage;
        lt_sfmx_in = lt_masked;
        mask_lts.push_back(lt_mask);
    }

    op sfmx(id++, op::kind::SoftMax, "softmax");
    sfmx.set_attr<int64_t>(op::attr::axis, -1);
    // "inf_as_zero": for rows where every input is -inf (e.g. a causal mask
    // row with no attended KV positions), treat the output as all-zero
    // instead of NaN. Matches the reference oneDNN SDPA examples and is
    // required to get correct behaviour on masked prefill rows; without it
    // NaNs propagate from masked rows into bmm2 and scramble the output.
    sfmx.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    sfmx.add_inputs({ lt_sfmx_in });
    sfmx.add_outputs({ lt_probs });

    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({ lt_probs, lt_v });
    bmm2.add_outputs({ lt_out });

    // ---- Graph ---------------------------------------------------------------
    dnnl::graph::graph sdpa_graph(dnnl::engine::kind::gpu);
    sdpa_graph.add_op(bmm1);
    sdpa_graph.add_op(div_op);
    if (add_op_ptr) {
        sdpa_graph.add_op(*add_op_ptr);
    }
    sdpa_graph.add_op(sfmx);
    sdpa_graph.add_op(bmm2);
    sdpa_graph.finalize();

    auto parts = sdpa_graph.get_partitions();
    if (parts.empty() || !parts[0].is_supported()) {
        throw std::runtime_error("oneDNN SDPA: no supported partition");
    }
    // Collect input ports in execution order: Q, K, scale, [mask], V
    std::vector<lt> in_ports;
    std::vector<lt> out_ports;
    in_ports.push_back(lt_q);
    in_ports.push_back(lt_k);
    in_ports.push_back(lt_scale);
    if (key.has_mask) {
        in_ports.insert(in_ports.end(), mask_lts.begin(), mask_lts.end());
    }
    in_ports.push_back(lt_v);
    out_ports.push_back(lt_out);

    auto cp = parts[0].compile(in_ports, out_ports, eng);

    // Allocate + populate the per-shape scalar divisor. Allocating here
    // (inside build_and_compile_sdpa, which the caller invokes under the
    // cache mutex) means the buffer is published as part of the entry only
    // after it is fully initialised — no write-site remains on the execute
    // path. The value is derived from `key.D`, not from any per-call scale
    // input: the oneDNN SDPA path requires scale == 1/sqrt(D) (enforced by
    // a runtime assertion at dispatch), so the divisor sqrt(D) is shape-
    // determined and a single buffer per compiled partition is correct for
    // all calls that reuse the partition.
    const size_t scale_bytes = (q_dt == dt::f32) ? sizeof(float) : sizeof(sycl::half);
    const int    device      = ggml_sycl_get_device_id_from_queue(*stream);

    ggml_sycl::alloc_request scale_req{};
    scale_req.queue                                    = stream;
    scale_req.device                                   = device;
    scale_req.size                                     = scale_bytes;
    scale_req.intent.role                              = ggml_sycl::alloc_role::CONTROL;
    scale_req.intent.category                          = ggml_sycl::runtime_category::CONTROL;
    scale_req.intent.cohort_id                         = "onednn_sdpa_scale";
    scale_req.intent.constraints.must_host_pinned      = true;
    scale_req.intent.constraints.use_pinned_pool       = true;
    scale_req.intent.constraints.require_host_usm_base = true;

    ggml_sycl::alloc_handle scale_alloc{};
    if (!ggml_sycl::unified_alloc(scale_req, &scale_alloc) || !scale_alloc.ptr) {
        throw std::runtime_error("oneDNN SDPA: failed to allocate USM scale buffer");
    }

    ggml_sycl::mem_handle scale_owner =
        ggml_sycl::mem_handle::from_owned_alloc(std::move(scale_alloc), GGML_LAYOUT_AOS);
    auto scale_r = scale_owner.resolve(device);
    if (!scale_r.ptr || scale_r.on_device) {
        throw std::runtime_error("oneDNN SDPA: scale buffer is not host-pinned USM");
    }

    void * scale_usm = scale_r.ptr;
    if (q_dt == dt::f32) {
        *static_cast<float *>(scale_usm) = sqrtf(static_cast<float>(key.D));
    } else {
        *static_cast<sycl::half *>(scale_usm) = static_cast<sycl::half>(sqrtf(static_cast<float>(key.D)));
    }

    build_result r;
    r.in_ports    = std::move(in_ports);
    r.out_ports   = std::move(out_ports);
    r.cp          = std::move(cp);
    r.scale_owner = std::move(scale_owner);
    r.scale_usm   = scale_usm;
    return r;
}

// -------------------------------------------------------------------
// Main dispatch entry
// -------------------------------------------------------------------
bool ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, const fattn_params & params) {
    dpct::queue_ptr stream = ctx.stream();

    const int H_q  = params.ne02;
    const int H_kv = params.ne12;
    const int D    = params.ne00;

    const bool                            multi_seq = (params.n_seqs > 1);
    const ggml_sycl_onednn_fa_layout_plan layout_plan =
        ggml_sycl_flash_attn_ext_onednn_plan(params, H_q, H_kv, params.kv_is_fp8, multi_seq);
    if (layout_plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT) {
        return false;
    }
    // oneDNN graph execute and the materialization allocation/repack are not
    // compatible with SYCL command graph recording. Follow the same policy as
    // DnnlMatMulWrapper: native FA handles recorded graphs.
    if (g_ggml_sycl_graph_recording) {
        return false;
    }

    // oneDNN graph SYCL execute requires device USM for ALL input tensors.
    // Any of Q, K, V may be host-pinned (sycl::malloc_host) under
    // GGML_SYCL_HOST_COMPUTE, GGML_SYCL_KV_HOST, host-resident weight
    // streaming, or warmup; skip oneDNN in any of those cases and fall
    // back to the XMX/TILE kernel which tolerates mixed host/device inputs.
    //
    // Fast check via VA range: Intel Arc device USM is allocated in the high
    // GPU VA space (addresses >= 0x800000000000) while host-pinned USM uses
    // normal user-space VAs. This avoids the expensive sycl::get_pointer_type()
    // call which syncs the L0 driver.
    static constexpr uintptr_t kDeviceVAThreshold = (uintptr_t) 1 << 47;  // ~128 TB
    if (reinterpret_cast<uintptr_t>(params.Q) < kDeviceVAThreshold) {
        return false;                                                     // Q is host/mmap — skip oneDNN
    }
    if (reinterpret_cast<uintptr_t>(params.K) < kDeviceVAThreshold ||
        reinterpret_cast<uintptr_t>(params.V) < kDeviceVAThreshold) {
        return false;  // K/V source is host/mmap — skip oneDNN and native FA will handle it.
    }

    fattn_params                        active_params = params;
    ggml_sycl_onednn_fa_materialized_kv materialized{};
    if (layout_plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED) {
        ggml_sycl_onednn_fa_materialization_desc desc{};
        if (!ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, H_q, H_kv, ctx.device, &desc) ||
            !desc.required) {
            if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
                fprintf(stderr,
                        "[SYCL] fattn: oneDNN MATERIALIZE_REQUIRED_BUT_UNAVAILABLE descriptor rejected "
                        "D=%d ne01=%d ne11=%d H_q=%d H_kv=%d k_nb1=%d k_nb2=%lld v_nb1=%d v_nb2=%lld\n",
                        D, params.ne01, params.ne11, H_q, H_kv, params.nb11, (long long) params.nb12, params.nb21,
                        (long long) params.nb22);
            }
            return false;
        }
        if (!ggml_sycl_flash_attn_ext_onednn_materialize_kv(desc, params, *stream, &materialized)) {
            if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
                fprintf(stderr,
                        "[SYCL] fattn: oneDNN MATERIALIZE_REQUIRED_BUT_UNAVAILABLE allocation_or_repack_failed "
                        "D=%d ne01=%d ne11=%d H_q=%d H_kv=%d bytes=%zu\n",
                        D, params.ne01, params.ne11, H_q, H_kv, desc.bytes_per_tensor);
            }
            return false;
        }
        auto k_resolved = materialized.K.resolve(ctx.device);
        auto v_resolved = materialized.V.resolve(ctx.device);
        if (!k_resolved || !v_resolved) {
            if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
                fprintf(stderr,
                        "[SYCL] fattn: oneDNN MATERIALIZE_REQUIRED_BUT_UNAVAILABLE handle_resolve_failed "
                        "device=%d\n",
                        ctx.device);
            }
            return false;
        }
        if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            fprintf(stderr, "[SYCL] fattn: oneDNN MATERIALIZED D=%d ne01=%d ne11=%d H_q=%d H_kv=%d bytes=%zu\n", D,
                    params.ne01, params.ne11, H_q, H_kv, desc.bytes_per_tensor);
        }
        active_params.K    = static_cast<const char *>(k_resolved.ptr);
        active_params.V    = static_cast<const char *>(v_resolved.ptr);
        active_params.nb11 = (int32_t) desc.k_target_nb1;
        active_params.nb12 = (int32_t) desc.k_target_nb2;
        active_params.nb13 = desc.k_target_nb3;
        active_params.nb21 = (int32_t) desc.v_target_nb1;
        active_params.nb22 = (int32_t) desc.v_target_nb2;
        active_params.nb23 = desc.v_target_nb3;
    }
    if (active_params.Q_type == GGML_TYPE_F32) {
        int32_t q_nb1 = 0;
        int64_t q_nb2 = 0;
        int64_t q_nb3 = 0;
        if (!ggml_sycl_flash_attn_ext_onednn_materialize_q_f32(active_params, ctx.device, *stream, &materialized,
                                                               &q_nb1, &q_nb2, &q_nb3)) {
            if (std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
                fprintf(stderr, "[SYCL] fattn: oneDNN Q_F32_MATERIALIZE_FAILED D=%d ne01=%d H_q=%d\n", D,
                        active_params.ne01, H_q);
            }
            return false;
        }
        auto q_resolved = materialized.Q.resolve(ctx.device);
        if (!q_resolved || !q_resolved.on_device) {
            return false;
        }
        active_params.Q      = static_cast<const char *>(q_resolved.ptr);
        active_params.Q_type = GGML_TYPE_F16;
        active_params.nb01   = q_nb1;
        active_params.nb02   = q_nb2;
        active_params.nb03   = q_nb3;
    }

    sdpa_shape_key key;
    key.device_id = ctx.device;
    key.D         = D;
    key.ncols     = active_params.ne01;
    key.ne11      = active_params.ne11;
    key.H_q       = H_q;
    key.H_kv      = H_kv;
    key.has_mask  = (active_params.mask != nullptr);
    key.is_gqa    = (H_q != H_kv);

    key.Q_type    = (int) active_params.Q_type;
    key.K_type    = (int) active_params.K_type;
    key.V_type    = (int) active_params.V_type;
    key.mask_type = (int) active_params.mask_type;

    key.q_nb1 = active_params.nb01;
    key.q_nb2 = active_params.nb02;
    key.q_nb3 = active_params.nb03;
    key.k_nb1 = active_params.nb11;
    key.k_nb2 = active_params.nb12;
    key.k_nb3 = active_params.nb13;
    key.v_nb1 = active_params.nb21;
    key.v_nb2 = active_params.nb22;
    key.v_nb3 = active_params.nb23;
    key.m_nb1 = key.has_mask ? active_params.nb31 : 0;
    key.m_nb2 = key.has_mask ? active_params.nb32 : 0;
    key.m_nb3 = key.has_mask ? active_params.nb33 : 0;

    sdpa_partition_cache * cache = get_or_create_cache(ctx);
    if (layout_plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT && std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
        fprintf(stderr, "[SYCL] fattn: oneDNN DIRECT D=%d ne01=%d ne11=%d H_q=%d H_kv=%d\n", D, active_params.ne01,
                active_params.ne11, H_q, H_kv);
    }

    // ------------------------------------------------------------------
    // Cache lookup + (if needed) compile — all under the per-cache mutex.
    // ------------------------------------------------------------------
    // We hold `cache->m` across the JIT compile in the miss path. This:
    //   - closes the TOCTOU window where two threads racing the same shape
    //     both compile, one insert wins, the other insert leaks work.
    //   - prevents rehashing/erasing `cache->hits` while another thread is
    //     still holding a reference to an entry (combined with shared_ptr
    //     ownership below, even a stray erase on the reader side would be
    //     safe, but we don't need to rely on that).
    //
    // Downside: other shape keys block during a slow compile. In practice
    // there are only a handful of unique shapes per workload (they warm the
    // cache during the first few prefill calls) so this serialisation is
    // bounded and amortised. The alternative (per-key promise/future) is
    // meaningfully more complex for a one-time cost.
    std::shared_ptr<sdpa_compiled_entry> entry;
    {
        // lock_guard, not unique_lock: we never unlock manually, move, or wait
        // on a condition_variable. lock_guard is lighter (no owns_lock flag)
        // and makes the "scope-bound lock" intent explicit.
        std::lock_guard<std::mutex> lock(cache->m);

        // Negative cache — previous compile for this shape failed; fall through fast.
        if (cache->negative.count(key)) {
            return false;
        }

        auto it = cache->hits.find(key);
        if (it != cache->hits.end()) {
            entry = it->second;  // shared_ptr copy; keeps the entry alive past unlock.
        } else {
            // Cache miss — compile while holding the lock. oneDNN JIT is slow
            // (10-100ms) only on the first call for a new shape; subsequent
            // calls hit the fast-path above.
            dnnl::engine eng = ctx.engine_dnnl(stream);
            try {
                build_result r   = build_and_compile_sdpa(key, eng, stream);
                auto         ent = std::make_shared<sdpa_compiled_entry>();
                ent->cp          = std::move(r.cp);
                ent->in_ports    = std::move(r.in_ports);
                ent->out_ports   = std::move(r.out_ports);
                // Transfer per-shape USM scratch ownership to the entry.
                // `scale_usm` remains only the oneDNN ABI pointer; the mem_handle
                // releases the allocation when the last shared_ptr drops.
                ent->scale_owner = std::move(r.scale_owner);
                ent->scale_usm   = r.scale_usm;
                r.scale_usm      = nullptr;
                // Store under the key, retain our own shared_ptr copy for use
                // below (entry survives any future cache mutation).
                cache->hits[key] = ent;
                entry            = std::move(ent);
            } catch (std::exception & e) {
                fprintf(stderr, "[SYCL] oneDNN SDPA compile failed for D=%d ncols=%d ne11=%d H_q=%d H_kv=%d: %s\n", D,
                        active_params.ne01, active_params.ne11, H_q, H_kv, e.what());
                cache->negative[key] = true;
                return false;
            }
        }
    }
    // ------------------------------------------------------------------
    // Lock released. `entry` is a shared_ptr — the compiled partition,
    // port lists, and per-shape `scale_usm` live as long as this local.
    // `scale_usm` was populated exactly once inside build_and_compile_sdpa
    // (under the cache mutex) and is never written again; the execute
    // path only READS it. No race.
    // ------------------------------------------------------------------

    // ---- Wrap USM pointers as dnnl::graph::tensor ----------------------------
    dnnl::engine eng = ctx.engine_dnnl(stream);

    // We build the graph with op::kind::Divide (to match oneDNN's canonical
    // SDPA pattern), so the scalar is the DIVISOR. The value stored in
    // `entry->scale_usm` is sqrt(D), pre-computed from `key.D` at partition
    // compile time. This path ASSERTS the invariant that ggml supplies
    // params.scale == 1/sqrt(D); any future model that deviates from it
    // needs a different fattn path (e.g. per-op scalar input) and should
    // fail loudly here rather than silently producing wrong outputs.
    //
    // Epsilon rationale: this assertion compares two f32 values
    // (`inv_scale = 1.0f/params.scale` and `sqrt_D = sqrtf(D)`). Drift
    // between them for a legitimate 1/sqrt(D) scale comes from reciprocal
    // rounding at ULP level (~1e-6 at target Ds 64/128/256), well below
    // 1e-3f. Wrong-formula bugs produce macroscopic diffs that this
    // tolerance catches easily (e.g. 1/D vs 1/sqrt(D) differ by ~11 at
    // D=128). The f16 narrowing happens elsewhere — at the STORE of
    // `*scale_usm` in build_and_compile_sdpa — and is separate from this
    // comparison.
    if (active_params.scale == 0.0f) {
        return false;  // zero scale would divide-by-zero; upstream never sends this
    }
    const float inv_scale  = 1.0f / active_params.scale;
    const float sqrt_D     = sqrtf(static_cast<float>(D));
    const float scale_diff = std::fabs(inv_scale - sqrt_D);
    GGML_ASSERT(scale_diff < 1e-3f &&
                "oneDNN SDPA: params.scale deviates from 1/sqrt(D) — model needs a different fattn path");

    const auto & in_ports  = entry->in_ports;
    const auto & out_ports = entry->out_ports;

    // in_ports order: Q, K, scale, [mask], V
    size_t port_idx = 0;

    dnnl::graph::tensor t_q(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(active_params.Q)));
    dnnl::graph::tensor t_k(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(active_params.K)));
    dnnl::graph::tensor t_scale(in_ports[port_idx++], eng, entry->scale_usm);

    dnnl::graph::tensor t_mask;
    if (key.has_mask) {
        t_mask = dnnl::graph::tensor(in_ports[port_idx++], eng,
                                     const_cast<void *>(static_cast<const void *>(active_params.mask)));
    }

    // V pointer: same element layout as K
    dnnl::graph::tensor t_v(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(active_params.V)));

    // Output is f32 directly into params.dst — no intermediate buffer needed.
    dnnl::graph::tensor t_out(out_ports[0], eng, active_params.dst);

    // Assemble input/output tensor lists
    std::vector<dnnl::graph::tensor> in_tensors;
    in_tensors.push_back(t_q);
    in_tensors.push_back(t_k);
    in_tensors.push_back(t_scale);
    if (key.has_mask) {
        in_tensors.push_back(t_mask);
    }
    in_tensors.push_back(t_v);

    std::vector<dnnl::graph::tensor> out_tensors = { t_out };

    try {
        dnnl::stream dnnl_stream = ctx.stream_dnnl(stream);
        dnnl::graph::sycl_interop::execute(entry->cp, dnnl_stream, in_tensors, out_tensors);
        if (materialized.Q.valid() || materialized.K.valid() || materialized.V.valid()) {
            std::vector<ggml_sycl::mem_handle> retained;
            retained.reserve(3);
            if (materialized.Q.valid()) {
                retained.push_back(std::move(materialized.Q));
            }
            if (materialized.K.valid()) {
                retained.push_back(std::move(materialized.K));
            }
            if (materialized.V.valid()) {
                retained.push_back(std::move(materialized.V));
            }
            sycl::event done = ggml_sycl_submit_marker<onednn_fa_materialized_release_marker_kernel>(*stream);
            ggml_sycl::retain_handles_until_event(std::move(retained), std::move(done));
        }
    } catch (std::exception & e) {
        fprintf(stderr, "[SYCL] oneDNN SDPA execute failed: %s\n", e.what());
        return false;
    }

    return true;
}

#endif  // GGML_SYCL_DNNL
