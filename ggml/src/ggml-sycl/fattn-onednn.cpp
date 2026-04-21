//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn-onednn.hpp"

#if GGML_SYCL_DNNL

#    include "common.hpp"
#    include "fattn-common.hpp"

#    include <atomic>
#    include <cstdio>
#    include <mutex>

using namespace dnnl::graph;
using lt       = logical_tensor;
using layout_t = lt::layout_type;
using dim_t    = lt::dim;
using dims_t   = lt::dims;
using dt       = lt::data_type;

// Per-context SDPA cache.  Stored via a raw pointer inside ggml_backend_sycl_context
// to avoid pulling graph headers into common.hpp.  Created on first use, freed at
// context teardown via ggml_sycl_sdpa_cache_destroy().
static std::mutex g_sdpa_cache_mutex;

static sdpa_partition_cache * get_or_create_cache(ggml_backend_sycl_context & ctx) {
    std::lock_guard<std::mutex> lock(g_sdpa_cache_mutex);
    if (!ctx.sdpa_cache) {
        ctx.sdpa_cache = new sdpa_partition_cache();
    }
    return static_cast<sdpa_partition_cache *>(ctx.sdpa_cache);
}

void ggml_sycl_sdpa_cache_destroy(void * ptr) {
    auto * cache = static_cast<sdpa_partition_cache *>(ptr);
    if (cache->scale_usm && cache->usm_queue) {
        sycl::free(cache->scale_usm, *cache->usm_queue);
    }
    delete cache;
}

// -------------------------------------------------------------------
// Eligibility gate
// -------------------------------------------------------------------
bool ggml_sycl_flash_attn_ext_onednn_eligible(const fattn_params & params,
                                              int                  H_q,
                                              int                  H_kv,
                                              bool                 kv_is_fp8,
                                              bool                 multi_seq) {
    if (params.sinks) {
        return false;  // attention sinks unsupported
    }
    if (params.logit_softcap != 0) {
        return false;  // softcap unsupported
    }
    if (kv_is_fp8) {
        return false;  // FP8 KV unsupported
    }
    if (multi_seq) {
        return false;  // multi-seq unsupported
    }
    if (params.ne00 > 512) {
        return false;  // D > 512 unsupported
    }
    if (params.ne11 <= 0) {
        return false;  // empty KV
    }
    // Only use for PP (ncols >= 8): TG (ncols=1-7) has unstable ne11 per step,
    // causing excessive JIT recompilation that dominates latency.
    if (params.ne01 < 8) {
        return false;
    }
    // Temporary diagnostic (removed in follow-up cleanup commit): log the first
    // three calls that reach this point so we can see which (Q_type, K_type,
    // V_type, mask_type) tuples the dtype gate below is rejecting. Placed
    // BEFORE the Q/K/V dtype checks so we see the real tensor dtypes.
    {
        static std::atomic<int> gate_dbg{ 0 };
        int                     n = gate_dbg.fetch_add(1);
        if (n < 3) {
            fprintf(stderr,
                    "[SYCL oneDNN] gate-trace call#%d  ne01=%d  Q_type=%d  K_type=%d  V_type=%d  has_mask=%d  "
                    "mask_type=%d\n",
                    n, params.ne01, (int) params.Q_type, (int) params.K_type, (int) params.V_type, params.mask ? 1 : 0,
                    params.mask ? (int) params.mask_type : -1);
        }
    }
    // The compiled partition is built with Q/K/V dtypes taken from params.*_type.
    // Scale and scratch buffer sizing assumes f16 Q (sizeof(half) scale slot),
    // and the existing path only supports f16 KV. Restrict to the well-tested
    // combination until the scale buffer is generalized.
    if (params.Q_type != GGML_TYPE_F16) {
        return false;
    }
    if (params.K_type != GGML_TYPE_F16) {
        return false;
    }
    if (params.V_type != GGML_TYPE_F16) {
        return false;
    }
    if (params.mask && params.mask_type != GGML_TYPE_F16 && params.mask_type != GGML_TYPE_F32) {
        return false;
    }
    (void) H_q;
    (void) H_kv;
    // Temporary diagnostic (remove once Fix A+B is validated on GPU):
    // log the first three gate hits so we can see which (ne01, ne11, mask)
    // shapes actually reach the oneDNN path from a real workload.
    static std::atomic<int> dbg_counter{ 0 };
    int                     n = dbg_counter.fetch_add(1);
    if (n < 3) {
        fprintf(stderr,
                "[SYCL oneDNN] eligible=true  call#%d  ne01=%d  ne11=%d  D=%d  H_q=%d  H_kv=%d  has_mask=%d  "
                "mask_type=%d\n",
                n, params.ne01, params.ne11, params.ne00, (int) params.ne02, (int) params.ne12, params.mask ? 1 : 0,
                params.mask ? (int) params.mask_type : -1);
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
struct build_result {
    std::vector<lt>                 in_ports;
    std::vector<lt>                 out_ports;
    dnnl::graph::compiled_partition cp;
};

static build_result build_and_compile_sdpa(const sdpa_shape_key & key, const dnnl::engine & eng) {
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

        // Score and output remain contiguous intermediates/dense dst.
        score_strides = { H_q * ncols * ne11, ncols * ne11, ne11, 1 };
        out_strides   = { H_q * ncols * D, ncols * D, D, 1 };
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
        // Dense f32 output layout (batch, H_q, ncols, D) viewed 5-D:
        // (batch, H_kv, N_rep, ncols, D) with h_q = kv_head*N_rep + rep
        out_strides   = { (dim_t) (H_kv * N_rep * ncols * D), (dim_t) (N_rep * ncols * D), (dim_t) (ncols * D),
                          (dim_t) (D), (dim_t) (1) };
    }

    // ---- Scale tensor (scalar, same dtype as Q) ---------------------------
    const dims_t scale_dims    = { 1 };
    const dims_t scale_strides = { 1 };

    // ---- Mask tensor -------------------------------------------------------
    // ggml mask shape: (ne30=ne11, ne31=ncols_padded, ne32=H_or_1, ne33=batch).
    // The row dimension is padded (GGML_KQ_MASK_PAD) so the physical row stride
    // nb31 is != ncols*elem_size. Use the real tensor byte strides.
    const dim_t m_s1 = key.has_mask ? static_cast<dim_t>(key.m_nb1 / (int64_t) m_esz) : 1;
    const dim_t m_s2 = key.has_mask ? static_cast<dim_t>(key.m_nb2 / (int64_t) m_esz) : 1;
    const dim_t m_s3 = key.has_mask ? static_cast<dim_t>(key.m_nb3 / (int64_t) m_esz) : 1;

    dims_t mask_dims, mask_strides;
    if (!is_gqa) {
        // (batch, heads_broadcast=1, ncols, ne11). Broadcast along head axis
        // by using the batch stride (mask has no per-head dim for non-GQA ATM).
        mask_dims    = { batch, 1, ncols, ne11 };
        mask_strides = { m_s3, m_s3, m_s1, 1 };
    } else {
        // (batch, H_kv=1, N_rep=1, ncols, ne11): same broadcast trick — both
        // head axes have extent 1, so stride is irrelevant beyond being valid.
        mask_dims    = { batch, 1, 1, ncols, ne11 };
        mask_strides = { m_s3, m_s3, m_s3, m_s1, 1 };
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

    build_result r;
    r.in_ports  = std::move(in_ports);
    r.out_ports = std::move(out_ports);
    r.cp        = std::move(cp);
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

    // oneDNN graph SYCL execute requires device USM for all input tensors.
    // K and V may be host-pinned (sycl::malloc_host) during warmup or when
    // weights are host-resident. Skip oneDNN in that case.
    //
    // Fast check via VA range: Intel Arc device USM is allocated in the high GPU VA space
    // (addresses >= 0x800000000000) while host-pinned USM uses normal user-space VAs.
    // This avoids the expensive sycl::get_pointer_type() call (which syncs the L0 driver).
    static constexpr uintptr_t kDeviceVAThreshold = (uintptr_t) 1 << 47;  // ~128 TB
    if (reinterpret_cast<uintptr_t>(params.K) < kDeviceVAThreshold) {
        return false;  // K is host/mmap — skip oneDNN (falls back to XMX/TILE kernel)
    }

    sdpa_shape_key key;
    key.device_id = ctx.device;
    key.D         = D;
    key.ncols     = params.ne01;
    key.ne11      = params.ne11;
    key.H_q       = H_q;
    key.H_kv      = H_kv;
    key.has_mask  = (params.mask != nullptr);
    key.is_gqa    = (H_q != H_kv);

    key.Q_type    = (int) params.Q_type;
    key.K_type    = (int) params.K_type;
    key.V_type    = (int) params.V_type;
    key.mask_type = (int) params.mask_type;

    key.q_nb1 = params.nb01;
    key.q_nb2 = params.nb02;
    key.q_nb3 = params.nb03;
    key.k_nb1 = params.nb11;
    key.k_nb2 = params.nb12;
    key.k_nb3 = params.nb13;
    key.v_nb1 = params.nb21;
    key.v_nb2 = params.nb22;
    key.v_nb3 = params.nb23;
    key.m_nb1 = key.has_mask ? params.nb31 : 0;
    key.m_nb2 = key.has_mask ? params.nb32 : 0;
    key.m_nb3 = key.has_mask ? params.nb33 : 0;

    sdpa_partition_cache * cache = get_or_create_cache(ctx);

    // Fast-path: negative cache (compile previously failed for this shape)
    {
        std::lock_guard<std::mutex> lock(g_sdpa_cache_mutex);
        if (cache->negative.count(key)) {
            return false;
        }
    }

    // Get or compile partition
    sdpa_compiled_entry * entry = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_sdpa_cache_mutex);
        auto                        it = cache->hits.find(key);
        if (it != cache->hits.end()) {
            entry = &it->second;
        }
    }

    if (!entry) {
        // Cache miss — compile (can be slow on first call, fast after JIT cache warms)
        dnnl::engine eng = ctx.engine_dnnl(stream);
        try {
            build_result r = build_and_compile_sdpa(key, eng);
            {
                std::lock_guard<std::mutex> lock(g_sdpa_cache_mutex);
                auto &                      e = cache->hits[key];
                e.cp                          = std::move(r.cp);
                e.in_ports                    = std::move(r.in_ports);
                e.out_ports                   = std::move(r.out_ports);
                entry                         = &cache->hits[key];
            }
        } catch (std::exception & e) {
            fprintf(stderr, "[SYCL] oneDNN SDPA compile failed for D=%d ncols=%d ne11=%d H_q=%d H_kv=%d: %s\n", D,
                    params.ne01, params.ne11, H_q, H_kv, e.what());
            std::lock_guard<std::mutex> lock(g_sdpa_cache_mutex);
            cache->negative[key] = true;
            return false;
        }
    }

    // oneDNN graph execute is NOT compatible with SYCL command graph recording.
    // Follow the same pattern as DnnlMatMulWrapper in dnnl-ops.hpp: skip during recording.
    if (g_ggml_sycl_graph_recording) {
        return false;
    }

    // ---- Wrap USM pointers as dnnl::graph::tensor ----------------------------
    dnnl::engine eng = ctx.engine_dnnl(stream);

    // Scale is a scalar f16.  oneDNN SYCL execute reads it from the GPU side,
    // so it must be USM-accessible.  Allocate once as sycl::malloc_host (pinned,
    // GPU-accessible via PCIe zero-copy) and reuse across calls.
    if (!cache->scale_usm) {
        cache->usm_queue = stream;
        cache->scale_usm = static_cast<sycl::half *>(sycl::malloc_host(sizeof(sycl::half), *stream));
        GGML_ASSERT(cache->scale_usm && "oneDNN SDPA: failed to allocate USM scale buffer");
    }
    // We now build the graph with op::kind::Divide (to match oneDNN's canonical
    // SDPA pattern), so the scalar must be the DIVISOR. ggml supplies
    // params.scale = 1/sqrt(D); invert it here to get sqrt(D) for the divide.
    // Guard against the (very unlikely) params.scale == 0 case with a passthrough
    // of 1.0 — upstream never feeds zero scale to flash-attn, but the div-by-zero
    // would be catastrophic if it happened.
    const float divisor = (params.scale != 0.0f) ? (1.0f / params.scale) : 1.0f;
    *cache->scale_usm   = static_cast<sycl::half>(divisor);

    const auto & in_ports  = entry->in_ports;
    const auto & out_ports = entry->out_ports;

    // in_ports order: Q, K, scale, [mask], V
    size_t port_idx = 0;

    dnnl::graph::tensor t_q(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.Q)));
    dnnl::graph::tensor t_k(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.K)));
    dnnl::graph::tensor t_scale(in_ports[port_idx++], eng, cache->scale_usm);

    dnnl::graph::tensor t_mask;
    if (key.has_mask) {
        t_mask =
            dnnl::graph::tensor(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.mask)));
    }

    // V pointer: same element layout as K
    dnnl::graph::tensor t_v(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.V)));

    // Output is f32 directly into params.dst — no intermediate buffer needed.
    dnnl::graph::tensor t_out(out_ports[0], eng, params.dst);

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

    // ---- Execute via SYCL interop API ----------------------------------------
    // Use sycl_interop::execute (not the member execute) so oneDNN properly
    // integrates with the SYCL queue's dependency chain and returns a SYCL event.
    // The member execute() uses the generic C API which may not sequence correctly
    // for SYCL streams, causing temporary_scratchpad_t destructor throws.
    try {
        dnnl::stream dnnl_stream = ctx.stream_dnnl(stream);
        dnnl::graph::sycl_interop::execute(entry->cp, dnnl_stream, in_tensors, out_tensors);
    } catch (std::exception & e) {
        fprintf(stderr, "[SYCL] oneDNN SDPA execute failed: %s\n", e.what());
        return false;
    }

    return true;
}

#endif  // GGML_SYCL_DNNL
