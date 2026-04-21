//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn-onednn.hpp"

#if GGML_SYCL_DNNL

#include "fattn-common.hpp"
#include "common.hpp"

#include <cstdio>
#include <mutex>

using namespace dnnl::graph;
using lt         = logical_tensor;
using layout_t   = lt::layout_type;
using dim_t      = lt::dim;
using dims_t     = lt::dims;
using dt         = lt::data_type;

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
                                               int                   H_q,
                                               int                   H_kv,
                                               bool                  kv_is_fp8,
                                               bool                  multi_seq) {
    if (params.sinks)              return false;  // attention sinks unsupported
    if (params.logit_softcap != 0) return false;  // softcap unsupported
    if (kv_is_fp8)                 return false;  // FP8 KV unsupported
    if (multi_seq)                 return false;  // multi-seq unsupported
    if (params.ne00 > 512)         return false;  // D > 512 unsupported
    if (params.ne11 <= 0)          return false;  // empty KV
    // Only use for PP (ncols >= 8): TG (ncols=1-7) has unstable ne11 per step,
    // causing excessive JIT recompilation that dominates latency.
    if (params.ne01 < 8)           return false;
    (void)H_q; (void)H_kv;
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
    std::vector<lt> in_ports;
    std::vector<lt> out_ports;
    dnnl::graph::compiled_partition cp;
};

static build_result build_and_compile_sdpa(const sdpa_shape_key & key,
                                            const dnnl::engine &    eng) {
    const int batch  = 1;
    const int D      = key.D;
    const int ncols  = key.ncols;
    const int ne11   = key.ne11;
    const int H_q    = key.H_q;
    const int H_kv   = key.H_kv;
    const int N_rep  = (H_kv > 0) ? (H_q / H_kv) : 1;
    const bool is_gqa = key.is_gqa;

    // Logical tensor IDs must be unique across the entire graph.
    size_t id = 0;

    // ---- Q / K / V shapes -------------------------------------------------
    // ggml layout: K[ne10=D, ne11=S, ne12=H_kv, ne13=batch] (row-major)
    // oneDNN wants (batch, H, S, D) contiguous — same memory, transposed dim interpretation.
    // We express this via strides derived from the ggml row-major layout.
    //
    // ggml strides (element units, elem_size = sizeof(half)):
    //   stride[0] = 1           (D dimension — innermost)
    //   stride[1] = D           (S dimension)
    //   stride[2] = D * ne11    (H dimension)
    //   stride[3] = D*ne11*H_kv (batch)
    //
    // oneDNN logical dims: (batch=1, H, S, D) → strides (D*S*H, D*S, D, 1)
    // This maps to: dim0=batch→stride[3], dim1=H→stride[2], dim2=S→stride[1], dim3=D→stride[0]
    // which equals the ggml layout strides exactly.

    dims_t q_dims, k_dims, v_dims, score_dims, out_dims;
    dims_t q_strides, k_strides, v_strides, score_strides, out_strides;

    if (!is_gqa) {
        // 4-D non-GQA: (batch, H, S, D)
        q_dims     = {batch, H_q,  ncols, D};
        k_dims     = {batch, H_kv, ne11,  D};
        v_dims     = {batch, H_kv, ne11,  D};
        score_dims = {batch, H_q,  ncols, ne11};
        out_dims   = {batch, H_q,  ncols, D};

        // Strides for (batch, H, S, D) contiguous → (H*S*D, S*D, D, 1)
        q_strides     = {H_q  * ncols * D,  ncols * D,  D,    1};
        k_strides     = {H_kv * ne11  * D,  ne11  * D,  D,    1};
        v_strides     = {H_kv * ne11  * D,  ne11  * D,  D,    1};
        score_strides = {H_q  * ncols * ne11, ncols * ne11, ne11, 1};
        out_strides   = {H_q  * ncols * D,  ncols * D,  D,    1};
    } else {
        // 5-D GQA: Q=(batch, H_kv, N_rep, ncols, D), K/V=(batch, H_kv, 1, ne11, D)
        q_dims     = {batch, H_kv, N_rep, ncols, D};
        k_dims     = {batch, H_kv, 1,     ne11,  D};
        v_dims     = {batch, H_kv, 1,     ne11,  D};
        score_dims = {batch, H_kv, N_rep, ncols, ne11};
        out_dims   = {batch, H_kv, N_rep, ncols, D};

        // Q strides: innermost D, then ncols*D per N_rep step, H_kv*N_rep*ncols*D per batch
        // Q is stored as (batch=1, H_q, ncols, D) in ggml.
        // We reinterpret as (1, H_kv, N_rep, ncols, D) — same memory.
        // stride[4]=1, stride[3]=D, stride[2]=ncols*D, stride[1]=N_rep*ncols*D, stride[0]=H_kv*N_rep*ncols*D
        q_strides = {
            (dim_t)(H_kv * N_rep * ncols * D),
            (dim_t)(N_rep * ncols * D),
            (dim_t)(ncols * D),
            (dim_t)(D),
            (dim_t)(1)
        };

        // K/V strides: (1, H_kv, 1, ne11, D) — K[ne10=D, ne11=S, ne12=H_kv, ne13=1]
        // dim5=1, dim4=D, dim3=ne11*D, dim2=ne11*D (head_rep stride = same as KV head stride),
        // dim1=ne11*D, dim0=H_kv*ne11*D
        k_strides = {
            (dim_t)(H_kv * ne11 * D),
            (dim_t)(ne11 * D),
            (dim_t)(ne11 * D),  // head_rep=1 → same stride as kv_head
            (dim_t)(D),
            (dim_t)(1)
        };
        v_strides = k_strides;

        score_strides = {
            (dim_t)(H_kv * N_rep * ncols * ne11),
            (dim_t)(N_rep * ncols * ne11),
            (dim_t)(ncols * ne11),
            (dim_t)(ne11),
            (dim_t)(1)
        };
        out_strides = q_strides;
    }

    // ---- Scale tensor (scalar f16) ----------------------------------------
    const dims_t scale_dims    = {1};
    const dims_t scale_strides = {1};

    // ---- Mask tensor -------------------------------------------------------
    // llama.cpp mask shape: (ne30=ne11, ne31=ncols, ne32=H_q_or_1, ne33=batch)
    // oneDNN Add broadcasts — we use (batch, 1, ncols, ne11) or 5-D equivalent.
    dims_t mask_dims, mask_strides;
    if (!is_gqa) {
        mask_dims    = {batch, 1, ncols, ne11};
        mask_strides = {ncols * ne11, ncols * ne11, ne11, 1};
    } else {
        mask_dims    = {batch, 1, 1, ncols, ne11};
        mask_strides = {ncols * ne11, ncols * ne11, ncols * ne11, ne11, 1};
    }

    // ---- Build logical tensors --------------------------------------------
    // Constructor: (id, dtype, dims, strides) — no layout_type arg for strided path
    lt lt_q     (id++, dt::f16, q_dims,     q_strides);
    lt lt_k     (id++, dt::f16, k_dims,     k_strides);
    lt lt_scale (id++, dt::f16, scale_dims, scale_strides);
    lt lt_v     (id++, dt::f16, v_dims,     v_strides);

    // Intermediate f32 tensors
    lt lt_score   (id++, dt::f32, score_dims, score_strides);
    lt lt_scaled  (id++, dt::f32, score_dims, score_strides);
    lt lt_masked  (id++, dt::f32, score_dims, score_strides);
    lt lt_probs   (id++, dt::f16, score_dims, score_strides);
    // Output is f32 to directly feed into params.dst (avoids intermediate f16 buffer + cast).
    lt lt_out     (id++, dt::f32, out_dims,   out_strides);

    // ---- Ops ---------------------------------------------------------------
    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({lt_q, lt_k});
    bmm1.add_outputs({lt_score});

    // params.scale = 1/sqrt(D) — multiply (not divide) to get scaled scores
    op div_op(id++, op::kind::Multiply, "scale_mul");
    div_op.add_inputs({lt_score, lt_scale});
    div_op.add_outputs({lt_scaled});

    // mask_add and softmax inputs depend on whether mask is present
    lt   lt_sfmx_in = lt_scaled;
    std::vector<lt> mask_lts;
    op * add_op_ptr = nullptr;
    op   add_op_storage(id++, op::kind::Add, "mask_add");

    if (key.has_mask) {
        lt lt_mask(id++, dt::f32, mask_dims, mask_strides);  // mask is f32 in ggml
        add_op_storage.add_inputs({lt_scaled, lt_mask});
        add_op_storage.add_outputs({lt_masked});
        add_op_ptr = &add_op_storage;
        lt_sfmx_in = lt_masked;
        mask_lts.push_back(lt_mask);
    }

    op sfmx(id++, op::kind::SoftMax, "softmax");
    sfmx.set_attr<int64_t>(op::attr::axis, -1);
    sfmx.add_inputs({lt_sfmx_in});
    sfmx.add_outputs({lt_probs});

    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({lt_probs, lt_v});
    bmm2.add_outputs({lt_out});

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
bool ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx,
                                      const fattn_params &          params) {
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
    static constexpr uintptr_t kDeviceVAThreshold = (uintptr_t)1 << 47;  // ~128 TB
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
        auto it = cache->hits.find(key);
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
                auto & e  = cache->hits[key];
                e.cp        = std::move(r.cp);
                e.in_ports  = std::move(r.in_ports);
                e.out_ports = std::move(r.out_ports);
                entry = &cache->hits[key];
            }
        } catch (std::exception & e) {
            fprintf(stderr, "[SYCL] oneDNN SDPA compile failed for D=%d ncols=%d ne11=%d H_q=%d H_kv=%d: %s\n",
                    D, params.ne01, params.ne11, H_q, H_kv, e.what());
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
        cache->usm_queue  = stream;
        cache->scale_usm  = static_cast<sycl::half *>(sycl::malloc_host(sizeof(sycl::half), *stream));
        GGML_ASSERT(cache->scale_usm && "oneDNN SDPA: failed to allocate USM scale buffer");
    }
    *cache->scale_usm = static_cast<sycl::half>(params.scale);

    const auto & in_ports  = entry->in_ports;
    const auto & out_ports = entry->out_ports;

    // in_ports order: Q, K, scale, [mask], V
    size_t port_idx = 0;

    dnnl::graph::tensor t_q(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.Q)));
    dnnl::graph::tensor t_k(in_ports[port_idx++], eng, const_cast<void *>(static_cast<const void *>(params.K)));
    dnnl::graph::tensor t_scale(in_ports[port_idx++], eng, cache->scale_usm);

    dnnl::graph::tensor t_mask;
    if (key.has_mask) {
        t_mask = dnnl::graph::tensor(in_ports[port_idx++], eng,
                                     const_cast<void *>(static_cast<const void *>(params.mask)));
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

    std::vector<dnnl::graph::tensor> out_tensors = {t_out};

    // ---- Execute via SYCL interop API ----------------------------------------
    // Use sycl_interop::execute (not the member execute) so oneDNN properly
    // integrates with the SYCL queue's dependency chain and returns a SYCL event.
    // The member execute() uses the generic C API which may not sequence correctly
    // for SYCL streams, causing temporary_scratchpad_t destructor throws.
    try {
        dnnl::stream dnnl_stream = ctx.stream_dnnl(stream);
        dnnl::graph::sycl_interop::execute(entry->cp, dnnl_stream,
                                           in_tensors, out_tensors);
    } catch (std::exception & e) {
        fprintf(stderr, "[SYCL] oneDNN SDPA execute failed: %s\n", e.what());
        return false;
    }

    return true;
}

#endif  // GGML_SYCL_DNNL
