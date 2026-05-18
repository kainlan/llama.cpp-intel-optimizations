#include "mmvq.hpp"

#include "common.hpp"
#include "convert.hpp"
#include "ggml-sycl-bench.hpp"
#include "ggml.h"
#include "quantize.hpp"
#include "quants.hpp"
#include "sycl-profiling.hpp"
#include "unified-kernel.hpp"  // For split barrier support
#include "vecdotq.hpp"

#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static thread_local mmvq_moe_dispatch_timing g_mmvq_moe_dispatch_timing;

void mmvq_moe_dispatch_timing_reset() {
    g_mmvq_moe_dispatch_timing = {};
}

mmvq_moe_dispatch_timing mmvq_moe_dispatch_timing_consume() {
    mmvq_moe_dispatch_timing timing = g_mmvq_moe_dispatch_timing;
    g_mmvq_moe_dispatch_timing      = {};
    return timing;
}

static bool mmvq_moe_tg_profile_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_MXFP4_TG_PROFILE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static double mmvq_sycl_event_duration_us(const sycl::event & ev) {
    try {
        const uint64_t start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end   = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        return end >= start ? static_cast<double>(end - start) * 1e-3 : -1.0;
    } catch (...) {
        return -1.0;
    }
}

struct mmvq_moe_tg_profile_accum {
    double  quant_us         = 0.0;
    double  batch_ids_us     = 0.0;
    double  kernel_us        = 0.0;
    double  artifact_us      = 0.0;
    double  gateup_glu_us    = 0.0;
    double  down_us          = 0.0;
    double  other_us         = 0.0;
    int64_t calls            = 0;
    int64_t entries          = 0;
    int64_t total_batches    = 0;
    int64_t soa_calls        = 0;
    int64_t coalesced_calls  = 0;
    int64_t aos_calls        = 0;
    int64_t gateup_glu_calls = 0;
    int64_t down_calls       = 0;
    int64_t other_calls      = 0;
};

static thread_local mmvq_moe_tg_profile_accum g_mmvq_moe_tg_profile;

enum class mmvq_moe_tg_profile_kind : uint8_t {
    OTHER      = 0,
    GATEUP_GLU = 1,
    DOWN       = 2,
};

static void mmvq_moe_tg_profile_record(layout_mode              layout,
                                       int64_t                  entries,
                                       int64_t                  total_batches,
                                       double                   quant_us,
                                       double                   batch_ids_us,
                                       double                   kernel_us,
                                       double                   artifact_us   = 0.0,
                                       mmvq_moe_tg_profile_kind kind          = mmvq_moe_tg_profile_kind::OTHER,
                                       int64_t                  logical_calls = 1) {
    auto & p = g_mmvq_moe_tg_profile;
    p.quant_us += quant_us;
    p.batch_ids_us += batch_ids_us;
    p.kernel_us += kernel_us;
    p.artifact_us += artifact_us;
    p.calls += logical_calls;
    p.entries += entries;
    p.total_batches += total_batches;
    const double role_us = quant_us + batch_ids_us + kernel_us;
    if (kind == mmvq_moe_tg_profile_kind::GATEUP_GLU) {
        p.gateup_glu_us += role_us;
        p.gateup_glu_calls += logical_calls;
    } else if (kind == mmvq_moe_tg_profile_kind::DOWN) {
        p.down_us += role_us;
        p.down_calls += logical_calls;
    } else {
        p.other_us += role_us;
        p.other_calls += logical_calls;
    }
    if (layout == GGML_LAYOUT_SOA) {
        p.soa_calls += logical_calls;
    } else if (layout == GGML_LAYOUT_COALESCED) {
        p.coalesced_calls += logical_calls;
    } else if (layout == GGML_LAYOUT_AOS) {
        p.aos_calls += logical_calls;
    }

    // GPT-OSS 20B has 72 MoE MUL_MAT_ID ops per token (24 layers x gate/up/down).
    // Dump on that cadence to keep TG profiling readable while still showing
    // token-level decode cost. This diagnostic intentionally synchronizes the queue.
    if (p.calls < 72 || p.calls % 72 != 0) {
        return;
    }

    const double total_us = p.quant_us + p.batch_ids_us + p.kernel_us + p.artifact_us;
    fprintf(stderr,
            "[MXFP4-MOE-TG-PROFILE] calls=%lld soa=%lld coalesced=%lld aos=%lld entries=%lld "
            "batches=%lld total=%.3f ms quant=%.3f ms artifact=%.3f ms batch_ids=%.3f ms kernel=%.3f ms "
            "gateup_glu=%.3f ms/%lld down=%.3f ms/%lld other=%.3f ms/%lld "
            "per_call total=%.3f us quant=%.3f us batch_ids=%.3f us kernel=%.3f us "
            "per_entry kernel=%.3f us\n",
            (long long) p.calls, (long long) p.soa_calls, (long long) p.coalesced_calls, (long long) p.aos_calls,
            (long long) p.entries, (long long) p.total_batches, total_us / 1000.0, p.quant_us / 1000.0,
            p.artifact_us / 1000.0, p.batch_ids_us / 1000.0, p.kernel_us / 1000.0, p.gateup_glu_us / 1000.0,
            (long long) p.gateup_glu_calls, p.down_us / 1000.0, (long long) p.down_calls, p.other_us / 1000.0,
            (long long) p.other_calls, total_us / static_cast<double>(p.calls),
            p.quant_us / static_cast<double>(p.calls), p.batch_ids_us / static_cast<double>(p.calls),
            p.kernel_us / static_cast<double>(p.calls),
            p.entries > 0 ? p.kernel_us / static_cast<double>(p.entries) : 0.0);
    p = {};
}

enum class mxfp4_moe_role : uint8_t {
    OTHER = 0,
    GATE  = 1,
    UP    = 2,
    DOWN  = 3,
};

static mxfp4_moe_role mxfp4_moe_role_from_name(const char * name) {
    if (!name) {
        return mxfp4_moe_role::OTHER;
    }
    if (std::strstr(name, "ffn_gate")) {
        return mxfp4_moe_role::GATE;
    }
    if (std::strstr(name, "ffn_up")) {
        return mxfp4_moe_role::UP;
    }
    if (std::strstr(name, "ffn_down")) {
        return mxfp4_moe_role::DOWN;
    }
    return mxfp4_moe_role::OTHER;
}

static int mxfp4_moe_layer_from_name(const char * name) {
    if (!name) {
        return -1;
    }
    const char * blk = std::strstr(name, "blk.");
    if (!blk) {
        return -1;
    }
    blk += 4;
    char * end = nullptr;
    long   v   = std::strtol(blk, &end, 10);
    if (end == blk || v < 0 || v > INT_MAX) {
        return -1;
    }
    return static_cast<int>(v);
}

struct mxfp4_moe_tg_reuse_cache {
    bool                    valid           = false;
    int                     device          = -1;
    int                     layer           = -1;
    mxfp4_moe_role          role            = mxfp4_moe_role::OTHER;
    const void *            src1_ptr        = nullptr;
    int64_t                 ne10            = 0;
    int64_t                 total_src1_rows = 0;
    int64_t                 ne10_padded     = 0;
    int64_t                 q8_1_row_size   = 0;
    size_t                  q8_bytes        = 0;
    layout_mode             layout          = GGML_LAYOUT_AOS;
    ggml_sycl::alloc_handle q8_alloc        = {};
    ggml_sycl::mem_handle   q8_handle       = {};
    ggml_sycl::mem_handle   src1_handle     = {};
};

static thread_local mxfp4_moe_tg_reuse_cache g_mxfp4_moe_tg_reuse;

static ggml_sycl::mem_handle mxfp4_moe_tensor_mem_handle(const ggml_tensor * tensor,
                                                         int                 device,
                                                         void *              ptr,
                                                         layout_mode         layout,
                                                         bool                on_device) {
    if (tensor && tensor->view_src == nullptr && tensor->extra && device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        auto * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);
        if (extra->data_handle[device].valid()) {
            return extra->data_handle[device];
        }
    }
    if (!ptr) {
        return {};
    }
    return ggml_sycl::mem_handle::from_chunk_ptr(ptr, device, layout, on_device);
}

static bool mxfp4_moe_tg_reuse_candidate(const ggml_tensor * src0,
                                         ggml_type           type,
                                         layout_mode         layout,
                                         int64_t             ne12,
                                         int64_t             total_src1_rows,
                                         mxfp4_moe_role *    role_out,
                                         int *               layer_out) {
    const mxfp4_moe_role role  = mxfp4_moe_role_from_name(src0 ? src0->name : nullptr);
    const int            layer = mxfp4_moe_layer_from_name(src0 ? src0->name : nullptr);
    if (role_out) {
        *role_out = role;
    }
    if (layer_out) {
        *layer_out = layer;
    }
    if (type != GGML_TYPE_MXFP4 || layout != GGML_LAYOUT_SOA || ne12 != 1 || layer < 0) {
        return false;
    }
    if (role == mxfp4_moe_role::GATE || role == mxfp4_moe_role::UP) {
        return total_src1_rows == 1;
    }
    if (role == mxfp4_moe_role::DOWN) {
        return total_src1_rows > 0;
    }
    return false;
}

static void mxfp4_moe_tg_reuse_invalidate() {
    g_mxfp4_moe_tg_reuse.valid       = false;
    g_mxfp4_moe_tg_reuse.role        = mxfp4_moe_role::OTHER;
    g_mxfp4_moe_tg_reuse.src1_handle = {};
}

static void * mxfp4_moe_tg_reuse_get_or_alloc_q8(sycl::queue * stream, int device, size_t required_size) {
    auto & cache = g_mxfp4_moe_tg_reuse;
    if (cache.q8_alloc.ptr && cache.q8_alloc.size >= required_size && cache.q8_alloc.device == device) {
        if (!cache.q8_handle.valid()) {
            cache.q8_handle = ggml_sycl::mem_handle::from_chunk_ptr(cache.q8_alloc.ptr, device, GGML_LAYOUT_SOA, true);
        }
        auto resolved = cache.q8_handle.resolve(device);
        return resolved ? resolved.ptr : cache.q8_alloc.ptr;
    }
    if (cache.q8_alloc.ptr) {
        cache.q8_handle = {};
        (void) ggml_sycl::unified_free(cache.q8_alloc);
        cache.q8_alloc = {};
    }
    ggml_sycl::alloc_request req{};
    req.queue                          = stream;
    req.device                         = device;
    req.size                           = required_size;
    req.intent.role                    = ggml_sycl::alloc_role::EXPERT_STAGING;
    req.intent.category                = ggml_sycl::runtime_category::STAGING;
    req.intent.constraints.must_device = true;
    if (!ggml_sycl::unified_alloc(req, &cache.q8_alloc)) {
        cache.q8_alloc  = {};
        cache.q8_handle = {};
        return nullptr;
    }
    cache.q8_handle = ggml_sycl::mem_handle::from_chunk_ptr(cache.q8_alloc.ptr, device, GGML_LAYOUT_SOA, true);
    auto resolved   = cache.q8_handle.resolve(device);
    return resolved ? resolved.ptr : cache.q8_alloc.ptr;
}

static bool mxfp4_moe_tg_reuse_can_use(int                           device,
                                       int                           layer,
                                       mxfp4_moe_role                role,
                                       const ggml_sycl::mem_handle & src1_handle,
                                       int64_t                       ne10,
                                       int64_t                       total_src1_rows,
                                       int64_t                       ne10_padded,
                                       int64_t                       q8_1_row_size,
                                       size_t                        q8_bytes,
                                       layout_mode                   layout) {
    const auto & cache = g_mxfp4_moe_tg_reuse;
    if (!cache.valid || !cache.q8_alloc.ptr) {
        return false;
    }
    const bool gate_up_pair = (cache.role == mxfp4_moe_role::GATE && role == mxfp4_moe_role::UP) ||
                              (cache.role == mxfp4_moe_role::UP && role == mxfp4_moe_role::GATE);
    const bool exact_role = cache.role == role && role == mxfp4_moe_role::DOWN;
    return (gate_up_pair || exact_role) && cache.device == device && cache.layer == layer &&
           cache.src1_handle == src1_handle && cache.ne10 == ne10 && cache.total_src1_rows == total_src1_rows &&
           cache.ne10_padded == ne10_padded && cache.q8_1_row_size == q8_1_row_size && cache.q8_bytes >= q8_bytes &&
           cache.layout == layout;
}

static void mxfp4_moe_tg_reuse_store(int                           device,
                                     int                           layer,
                                     mxfp4_moe_role                role,
                                     const ggml_sycl::mem_handle & src1_handle,
                                     int64_t                       ne10,
                                     int64_t                       total_src1_rows,
                                     int64_t                       ne10_padded,
                                     int64_t                       q8_1_row_size,
                                     size_t                        q8_bytes,
                                     layout_mode                   layout) {
    auto & cache          = g_mxfp4_moe_tg_reuse;
    cache.valid           = true;
    cache.device          = device;
    cache.layer           = layer;
    cache.role            = role;
    cache.src1_ptr        = src1_handle.resolve().ptr;
    cache.src1_handle     = src1_handle;
    cache.ne10            = ne10;
    cache.total_src1_rows = total_src1_rows;
    cache.ne10_padded     = ne10_padded;
    cache.q8_1_row_size   = q8_1_row_size;
    cache.q8_bytes        = q8_bytes;
    cache.layout          = layout;
}

static bool mxfp4_moe_tg_publish_q8_soa(sycl::queue *                 stream,
                                        int                           device,
                                        int                           layer,
                                        mxfp4_moe_role                role,
                                        const float *                 src,
                                        int64_t                       ne0,
                                        int64_t                       total_rows,
                                        double *                      quant_us_out        = nullptr,
                                        const ggml_sycl::mem_handle * src_handle_override = nullptr) {
    if (!stream || !src || device < 0 || layer < 0 || role != mxfp4_moe_role::DOWN || ne0 <= 0 || total_rows <= 0) {
        return false;
    }
    const int64_t ne0_padded  = GGML_PAD(ne0, QK8_1);
    const int64_t q8_row_size = ne0_padded * sizeof(block_q8_1) / QK8_1;
    const size_t  q8_bytes    = static_cast<size_t>(total_rows) * static_cast<size_t>(q8_row_size);
    void *        q8_buffer   = mxfp4_moe_tg_reuse_get_or_alloc_q8(stream, device, q8_bytes);
    if (!q8_buffer) {
        mxfp4_moe_tg_reuse_invalidate();
        return false;
    }

    const bool profile = mmvq_moe_tg_profile_enabled();
    if (profile) {
        stream->wait();
    }
    auto t_begin = std::chrono::high_resolution_clock::now();
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src, (char *) q8_buffer, ne0, total_rows, ne0_padded, stream);
    if (profile) {
        stream->wait();
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    if (quant_us_out) {
        *quant_us_out += std::chrono::duration<double, std::micro>(t_end - t_begin).count();
    }

    ggml_sycl::mem_handle src_handle;
    if (src_handle_override && src_handle_override->valid()) {
        src_handle        = *src_handle_override;
        auto src_resolved = src_handle.resolve(device);
        if (!src_resolved || src_resolved.ptr != src) {
            src_handle = {};
        }
    }
    if (!src_handle.valid()) {
        src_handle = ggml_sycl::mem_handle::from_chunk_ptr(const_cast<float *>(src), device, GGML_LAYOUT_AOS, true);
    }
    mxfp4_moe_tg_reuse_store(device, layer, role, src_handle, ne0, total_rows, ne0_padded, q8_row_size, q8_bytes,
                             GGML_LAYOUT_SOA);
    GGML_SYCL_DEBUG("[MOE-Q8-ARTIFACT] STORE role=down layer=%d src=%p rows=%lld ne0=%lld bytes=%zu q8=%p\n", layer,
                    (const void *) src, (long long) total_rows, (long long) ne0, q8_bytes, q8_buffer);
    return true;
}

// Returns true if the tensor's weights are host-resident (not in device VRAM).
// Simplified version for mmvq.cpp — ggml_backend_sycl_buffer_context is not
// accessible here, so we skip the tier-based check and rely on buffer host flag
// + USM pointer type query with try/catch for driver exception safety.
static bool ggml_sycl_is_host_resident_weight(const ggml_tensor * src0, sycl::queue * stream) {
    // Check 1: ggml backend buffer host flag
    if (src0->buffer && ggml_backend_buffer_is_host(src0->buffer)) {
        return true;
    }
    // Check 2: USM pointer type query (may return unknown on multi-device L0)
    if (ggml_sycl_host_data(src0) && stream) {
        try {
            const auto alloc = ggml_sycl_get_alloc_type(const_cast<void *>(ggml_sycl_host_data(src0)));
            return alloc != sycl::usm::alloc::device;
        } catch (...) {
            return true;  // Assume host for safety
        }
    }
    return false;
}

// Kernel name classes for VTune/profiler visibility
// Note: Using int instead of ggml_type because SYCL kernel names require fixed underlying types
template <int qtype> class mmvq_kernel_name;
template <int qtype> class mmvq_reorder_kernel_name;
template <int qtype> class mmvq_reorder_slm_kernel_name;
template <int qtype> class mmvq_coalesced_kernel_name;
template <int qtype> class mmvq_id_kernel_name;

// SoA MMVQ kernel template
// Note: total_nrows is the full tensor row count (ne01), used for SoA offset calculations
//       nrows is the number of rows to process (row_diff), used for bounds checking
//       row_low is the starting row offset for this slice (for split tensor support)
// This distinction is critical because SoA layout is created with full tensor dimensions,
// but MMVQ may process only a subset of rows starting at row_low.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx,
                                  const void * __restrict__ vy,
                                  float * __restrict__ dst,
                                  const int                ncols,
                                  const int                nrows,
                                  const int                total_nrows,
                                  const int                row_low,
                                  const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    // Use total_nrows (full tensor size) for SoA offset calculations, NOT nrows (slice size)
    const int     nblocks                     = total_nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        // For SoA layout, use global row index (row_low + row) for block indexing
        // because SoA offsets are calculated based on full tensor dimensions
        const int ibx = (row_low + row) * blocks_per_row + i;  // x block index (global)

        const auto          bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto          d_offset       = block_type::get_d_offset(total_nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int           iby            = i * block_type::block_to_q8_1_ratio();
        const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Multi-row reordered kernel with SLM Y-vector sharing
// All subgroups in the work-group share Y-vector cached in SLM
// Note: total_nrows is the full tensor row count (ne01), used for SoA offset calculations
//       row_low is the starting row offset for this slice (for split tensor support)
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder_slm(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int                ncols,
                                      const int                nrows,
                                      const int                total_nrows,
                                      const int                row_low,
                                      const sycl::nd_item<3> & nd_item,
                                      int8_t * __restrict__ slm_y_qs,
                                      sycl::half2 * __restrict__ slm_y_ds) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  lane_id      = sg.get_local_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    const int blocks_per_row = ncols / block_traits::qk;

    // Step 1: First subgroup loads Y-vector to SLM cooperatively
    // Y is in reordered format: quants at vy[0..ncols-1], ds at vy[ncols..]
    if (sg_id == 0) {
        // Load Y quants (int8) - ncols bytes total
        const int8_t * y_qs = (const int8_t *) vy;
        for (int i = lane_id; i < ncols; i += WARP_SIZE) {
            slm_y_qs[i] = y_qs[i];
        }

        // Load Y ds (half2) - blocks_per_row entries
        const sycl::half2 * y_ds = (const sycl::half2 *) ((const char *) vy + ncols);
        for (int i = lane_id; i < blocks_per_row; i += WARP_SIZE) {
            slm_y_ds[i] = y_ds[i];
        }
    }

    // Barrier: wait for Y to be loaded to SLM
    nd_item.barrier(sycl::access::fence_space::local_space);

    if (row >= nrows) {
        return;
    }

    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    // Use total_nrows (full tensor size) for SoA offset calculations, NOT nrows (slice size)
    const int     nblocks                     = total_nrows * blocks_per_row;

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = lane_id / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        // For SoA layout, use global row index (row_low + row) for block indexing
        // because SoA offsets are calculated based on full tensor dimensions
        const int ibx = (row_low + row) * blocks_per_row + i;  // x block index (global)

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(total_nrows, ncols, ibx);

        // Y block index that aligns with ibx
        const int           iby            = i * block_type::block_to_q8_1_ratio();
        // Use SLM-cached Y data instead of device memory
        const int8_t *      q8_1_quant_ptr = slm_y_qs + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr    = slm_y_ds + iby;

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (lane_id % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Warp-coalesced MMVQ kernel for Q4_0
// Tensor layout: within each tile, data is word-major instead of block-major
// Word w of block b is at: tile_offset + w * (TILE_BLOCKS*4) + b * 4
// This achieves 100% cache line utilization (vs 50% with strided access in standard reorder)
static void mul_mat_vec_q4_0_coalesced(const void * __restrict__ vx,  // Coalesced X weights
                                       const void * __restrict__ vy,  // Reordered Y activations
                                       float * __restrict__ dst,
                                       const int                ncols,
                                       const int                nrows,
                                       const sycl::nd_item<3> & nd_item) {
    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  lane_id      = sg.get_local_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK4_0;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;

    const int word_stride = TILE_BLOCKS * 4;

    // X base pointers (coalesced layout: quants first, then scales)
    // Quants: tiles_per_row * TILE_BLOCKS * 16 bytes per row = ncols/2 bytes
    const uint8_t * x_qs         = (const uint8_t *) vx;
    const int       x_row_stride = ncols / 2;  // bytes per row of quants

    // Scales are after all quants in the tensor
    const ggml_half * x_d = (const ggml_half *) ((const char *) vx + nrows * x_row_stride);

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t *      y_qs = (const int8_t *) vy;
    const sycl::half2 * y_ds = (const sycl::half2 *) ((const char *) vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES;

        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            // Get scale for this block (scales are NOT coalesced, remain block-sequential)
            const int   block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
            const float d         = x_d[block_idx];

            // Y block index and base offset
            const int y_block = tile * TILE_BLOCKS + block_in_tile;
            const int y_base  = y_block * QK8_1;

            for (int half = 0; half < 2; ++half) {
                // Coalesced load: word w of block b at offset w*word_stride + b*4
                // Thread loads 2 words (8 bytes total) per half
                const int word_base    = half * (2 * word_stride);
                const int word0_offset = word_base + block_in_tile * 4;                // word 0 or 2
                const int word1_offset = word_base + word_stride + block_in_tile * 4;  // word 1 or 3

                // Perfectly coalesced 4-byte loads
                const int v0 = *((const int *) (x_qs + tile_base + word0_offset));
                const int v1 = *((const int *) (x_qs + tile_base + word1_offset));

                // Load Y data matching the X half we're processing
                // For half=0: Y elements 0-3, 16-19, 4-7, 20-23
                // For half=1: Y elements 8-11, 24-27, 12-15, 28-31
                const int y_offset = half * 8;  // 0 or 8 (in terms of bytes)
                const int u0       = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);  // Y[0:3] or Y[8:11]
                const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);    // Y[16:19] or Y[24:27]
                const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);    // Y[4:7] or Y[12:15]
                const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);    // Y[20:23] or Y[28:31]

                // Extract nibbles and compute dp4a
                const int vi0_0 = (v0 >> 0) & 0x0F0F0F0F;  // low nibbles of word 0
                const int vi1_0 = (v0 >> 4) & 0x0F0F0F0F;  // high nibbles of word 0
                const int vi0_1 = (v1 >> 0) & 0x0F0F0F0F;  // low nibbles of word 1
                const int vi1_1 = (v1 >> 4) & 0x0F0F0F0F;  // high nibbles of word 1

                int sumi = 0;
                sumi     = dpct::dp4a(vi0_0, u0, sumi);
                sumi     = dpct::dp4a(vi1_0, u1, sumi);
                sumi     = dpct::dp4a(vi0_1, u2, sumi);
                sumi     = dpct::dp4a(vi1_1, u3, sumi);

                // Apply scales: result = d4 * (sumi * ds8.x - 4 * ds8.y)
                // The 4 comes from: 8 elements * (subtract 8 offset) / 16 = 4
                const sycl::half2  ds8  = y_ds[y_block];
                const sycl::float2 ds8f = ds8.convert<float, sycl::rounding_mode::automatic>();
                partial_sum += d * (sumi * ds8f.x() - 4.0f * ds8f.y());
            }
        }
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// Q6_K coalesced kernel - processes tile blocks with word-major layout
// =============================================================================
// Q6_K VARIABLE TILE COALESCED KERNEL
// Handles arbitrary block counts using power-of-2 tile decomposition
// Each work-group processes one row, with multiple warps handling different tiles
// Shared memory reduction combines partial sums from all tiles
// =============================================================================
static void mul_mat_vec_q6_k_variable_tile(const void * __restrict__ vx,
                                           const void * __restrict__ vy,
                                           float * __restrict__ dst,
                                           const int ncols,
                                           const int nrows,
                                           const int total_nrows,
                                           const int row_low,
                                           const int blocks_per_row,
                                           const int num_tiles,
                                           const int row_quants_bytes,
                                           float * __restrict__ shared_partials,
                                           const sycl::nd_item<3> & nd_item) {
    const auto sg         = nd_item.get_sub_group();
    const int  warp_id    = nd_item.get_local_id(2) / WARP_SIZE;
    const int  lane_id    = sg.get_local_linear_id();
    const int  local_row  = nd_item.get_group(2);  // Row within this slice
    const int  global_row = row_low + local_row;   // Row in full tensor

    if (local_row >= nrows) {
        return;
    }

    // X base pointers
    const uint8_t * x_base = (const uint8_t *) vx;

    // Y pointers
    const int8_t *      y_qs = (const int8_t *) vy;
    const sycl::half2 * y_ds = (const sycl::half2 *) ((const char *) vy + ncols);

    // D values at end of tensor - use total_nrows since d values are at end of FULL tensor
    const ggml_half * x_d = (const ggml_half *) (x_base + total_nrows * row_quants_bytes);

    float partial_sum = 0.0f;

    // Only warps assigned to tiles do actual work
    // Extra warps will contribute 0 to the reduction
    if (warp_id < num_tiles) {
        // Get this warp's tile info using precomputed values
        // We can't call tile_size_at in kernel, so we recompute locally
        int tile_size   = 0;
        int tile_offset = 0;
        {
            int remaining      = blocks_per_row;
            int current_offset = 0;
            for (int t = 0; t <= warp_id && remaining > 0; t++) {
                int ts = 1;
                while (ts * 2 <= remaining && ts < 32) {
                    ts *= 2;
                }
                if (t == warp_id) {
                    tile_size   = ts;
                    tile_offset = current_offset;
                }
                current_offset += ts;
                remaining -= ts;
            }
        }

        // Compute byte offset to this tile within the row
        int tile_byte_offset = 0;
        {
            int remaining      = blocks_per_row;
            int current_offset = 0;
            for (int t = 0; t < warp_id && remaining > 0; t++) {
                int ts = 1;
                while (ts * 2 <= remaining && ts < 32) {
                    ts *= 2;
                }
                tile_byte_offset += ts * (128 + 64 + 16);
                current_offset += ts;
                remaining -= ts;
            }
        }

        const int word_plane_stride = tile_size * 4;

        // X pointers - use global_row for accessing the full tensor data
        const uint8_t * tile_base = x_base + global_row * row_quants_bytes + tile_byte_offset;

        // Tile layout: [ql: tile_size * 128][qh: tile_size * 64][scales: tile_size * 16]
        const uint8_t * tile_ql = tile_base;
        const uint8_t * tile_qh = tile_ql + tile_size * 128;
        const int8_t *  tile_sc = (const int8_t *) (tile_qh + tile_size * 64);

        // Only active lanes process blocks
        if (lane_id < tile_size) {
            const int   block_in_tile    = lane_id;
            const int   global_block_idx = global_row * blocks_per_row + tile_offset + block_in_tile;
            const float d                = x_d[global_block_idx];

            const int y_block_base = (tile_offset + block_in_tile) * 8;

            for (int iqs = 0; iqs < QI6_K; ++iqs) {
                const int vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
                const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
                const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
                const int u_index      = iqs % QI8_1;

                const int ql_offset   = iqs * word_plane_stride + block_in_tile * 4;
                const int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
                const int qh_offset   = qh_word_idx * word_plane_stride + block_in_tile * 4;
                const int vl          = *((const int *) (tile_ql + ql_offset));
                const int vh_raw      = *((const int *) (tile_qh + qh_offset));
                const int vh          = vh_raw >> vh_shift;

                const int    sc_idx0    = scale_offset;
                const int    sc_idx1    = scale_offset + 4;
                const int    sc_word0   = sc_idx0 / 4;
                const int    sc_byte0   = sc_idx0 % 4;
                const int    sc_word1   = sc_idx1 / 4;
                const int    sc_byte1   = sc_idx1 % 4;
                const int    sc_offset0 = sc_word0 * word_plane_stride + block_in_tile * 4 + sc_byte0;
                const int    sc_offset1 = sc_word1 * word_plane_stride + block_in_tile * 4 + sc_byte1;
                const int8_t sc0        = tile_sc[sc_offset0];
                const int8_t sc1        = tile_sc[sc_offset1];

                const int          y_block0 = y_block_base + bq8_offset;
                const int          y_block1 = y_block0 + 2;
                const int          u0       = get_int_from_int8_aligned(y_qs + y_block0 * QK8_1, u_index);
                const int          u1       = get_int_from_int8_aligned(y_qs + y_block1 * QK8_1, u_index);
                const sycl::float2 ds0f     = y_ds[y_block0].convert<float, sycl::rounding_mode::automatic>();
                const sycl::float2 ds1f     = y_ds[y_block1].convert<float, sycl::rounding_mode::automatic>();
                const float        d8_0     = ds0f.x();
                const float        d8_1     = ds1f.x();

                float     sumf = 0.0f;
                const int vil0 = (vl >> 0) & 0x0F0F0F0F;
                const int vih0 = ((vh >> 0) << 4) & 0x30303030;
                const int vi0  = dpct::vectorized_binary<sycl::char4>((vil0 | vih0), 0x20202020, dpct::sub_sat());
                sumf += d8_0 * (dpct::dp4a(vi0, u0, 0) * sc0);

                const int vil1 = (vl >> 4) & 0x0F0F0F0F;
                const int vih1 = ((vh >> 4) << 4) & 0x30303030;
                const int vi1  = dpct::vectorized_binary<sycl::char4>((vil1 | vih1), 0x20202020, dpct::sub_sat());
                sumf += d8_1 * (dpct::dp4a(vi1, u1, 0) * sc1);

                partial_sum += d * sumf;
            }
        }
    }

    // Warp-level reduction - ALL warps participate (extra warps contribute 0)
    float warp_sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());

    // Write to shared memory - ALL warps write (extra warps write 0)
    if (lane_id == 0) {
        shared_partials[warp_id] = warp_sum;
    }

    // Split barrier for proper visibility across sub-groups on Intel Arc
    // Arrive is non-blocking, allowing prefetch/other work before wait
    split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
    // Future optimization: prefetch next layer's weights here
    split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

    // First warp reduces all partial sums
    if (warp_id == 0) {
        float val       = (lane_id < num_tiles) ? shared_partials[lane_id] : 0.0f;
        float final_sum = sycl::reduce_over_group(sg, val, sycl::plus<float>());
        if (lane_id == 0) {
            dst[local_row] = final_sum;  // Output to slice-relative position
        }
    }
}

// Dispatch function for Q6_K variable tile kernel
// total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
// row_low is the starting row offset for this slice
static void variable_tile_mul_mat_vec_q6_k_q8_1_sycl(const void *  vx,
                                                     const void *  vy,
                                                     float *       dst,
                                                     const int     ncols,
                                                     const int     nrows,
                                                     const int     total_nrows,
                                                     const int     row_low,
                                                     sycl::queue & stream) {
    const int blocks_per_row  = ncols / QK_K;
    const int num_tiles       = tile_count(blocks_per_row);
    const int threads_per_row = num_tiles * WARP_SIZE;

    // Compute row stride
    int row_quants_bytes = 0;
    {
        int remaining = blocks_per_row;
        while (remaining > 0) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            row_quants_bytes += ts * (128 + 64 + 16);
            remaining -= ts;
        }
    }

    GGML_SYCL_DEBUG(
        "[MMVQ] Q6_K variable tile: blocks_per_row=%d, num_tiles=%d, threads=%d, total_nrows=%d, row_low=%d\n",
        blocks_per_row, num_tiles, threads_per_row, total_nrows, row_low);

    sycl::range<3> grid(1, 1, nrows);
    sycl::range<3> block(1, 1, threads_per_row);

    stream.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_partials(sycl::range<1>(num_tiles), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> nd_item) [[intel::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q6_k_variable_tile(vx, vy, dst, ncols, nrows, total_nrows, row_low,
                                                            blocks_per_row, num_tiles, row_quants_bytes,
                                                            shared_partials.get_pointer(), nd_item);
                         });
    });
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx,
                          const void * __restrict__ vy,
                          float * __restrict__ dst,
                          const int                ncols,
                          const int                nrows,
                          const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    // Hoist invariant iqs calculation outside the loop
    constexpr int qi_div_vdr = qi / vdr;
    const int     lane_id    = item_ct1.get_local_id(2);
    const int     base_iqs   = vdr * (lane_id % qi_div_vdr);
    const int     row_offset = row * blocks_per_row;

    // 4-way accumulator for better ILP (matches Xe2 4-cycle FMA latency)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y[iby0], iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y[iby1], iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y[iby2], iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y[iby3], iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // Combine accumulators (tree reduction for fewer dependencies)
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Use subgroup reduce for final reduction (more efficient than manual XOR)
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

// MoE-aware kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
// For MUL_MAT_ID: reads ids[iid1][id] to determine which expert weights to use
// This allows GPU-side expert routing without host sync
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_id(const void * __restrict__ vx,
                             const void * const * __restrict__ expert_ptrs,
                             const void * __restrict__ vy,
                             float * __restrict__ dst,
                             const int32_t * __restrict__ ids,
                             const int                ncols,
                             const int                nrows_per_expert,
                             const int                n_ids,
                             const int                n_tokens,
                             const int                ne11,
                             const int64_t            stride_expert_x,
                             const int64_t            ids_nb0,
                             const int64_t            ids_nb1,
                             const int64_t            nb11,
                             const int64_t            nb12,
                             const int64_t            nb1,
                             const int64_t            nb2,
                             const sycl::nd_item<3> & item_ct1) {
    // batch_idx from block.y dimension (linearized over id * n_tokens + iid1)
    const int batch_idx = item_ct1.get_group(1);
    // row within expert from block.z dimension
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - row-major by id
    const int id   = batch_idx / n_tokens;       // Expert selection index
    const int iid1 = batch_idx - id * n_tokens;  // Token position

    int32_t expert_id = -1;
    if (ids) {
        // Read expert ID from ids tensor using proper 2D indexing
        expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    }

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1  = id;
    const int64_t i2  = iid1;

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    assert(blocks_per_warp > 0);

    // Expert weights: from pointer table when provided, else stride-based
    const block_q_t * x = nullptr;
    if (expert_ptrs) {
        if (ids) {
            x = static_cast<const block_q_t *>(expert_ptrs[expert_id]);
        } else {
            x = static_cast<const block_q_t *>(expert_ptrs[batch_idx]);
        }
    } else if (expert_id >= 0) {
        x = (const block_q_t *) ((const char *) vx + expert_id * stride_expert_x);
    }
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char *) vy + i11 * nb11 + i12 * nb12);

    // Hoist invariant calculations outside the loop
    constexpr int qi_div_vdr = qi / vdr;
    const int     lane_id    = item_ct1.get_local_id(2);
    const int     base_iqs   = vdr * (lane_id % qi_div_vdr);
    const int     row_offset = row * blocks_per_row;

    // 4-way accumulator for better ILP (matches Xe2 4-cycle FMA latency)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y[iby0], iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y[iby1], iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y[iby2], iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y[iby3], iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // Combine accumulators (tree reduction for fewer dependencies)
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Use subgroup reduce for final reduction (more efficient than manual XOR)
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float *) ((char *) dst + i1 * nb1 + i2 * nb2);
        dst_out[row]    = tmp;
    }
}

// Multi-row MMVQ kernel: processes multiple output rows per work-group
// Shares Y-vector in SLM across all rows to reduce memory bandwidth
// Expected +15-25% improvement for token generation
template <int qk,
          int qi,
          typename block_q_t,
          int vdr,
          float (*vec_dot_q_slm)(const void *, const int *, const sycl::half2 *, int, int, const int &),
          int nrows_per_wg>
static void mul_mat_vec_q_multirow(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int                ncols,
                                   const int                nrows,
                                   const sycl::nd_item<3> & item_ct1,
                                   int * __restrict__ slm_y_qs,
                                   sycl::half2 * __restrict__ slm_y_ds) {
    // Work-group layout: (1, nrows_per_wg, WARP_SIZE)
    // Each warp handles one row, all warps share Y-vector in SLM
    const int local_row = item_ct1.get_local_id(1);           // Which row within work-group (0 to nrows_per_wg-1)
    const int lane_id   = item_ct1.get_local_id(2);           // Thread within warp (0 to 31)
    const int wg_idx    = item_ct1.get_group(2);              // Work-group index
    const int row       = wg_idx * nrows_per_wg + local_row;  // Global row index

    const int blocks_per_row = ncols / qk;

    // Step 1: First warp (local_row == 0) loads Y-vector to SLM
    // All threads in the first warp cooperatively load Y data
    if (local_row == 0) {
        const block_q8_1 * y = (const block_q8_1 *) vy;

        // Each thread loads its share of blocks
        for (int blk = lane_id; blk < blocks_per_row; blk += WARP_SIZE) {
            // Load 8 ints (32 bytes) of quantized data per block
            const int slm_offset = blk * MMVQ_SLM_Y_QS_STRIDE;
#pragma unroll
            for (int j = 0; j < QI8_1; ++j) {
                slm_y_qs[slm_offset + j] = get_int_from_int8_aligned(y[blk].qs, j);
            }
            // Load ds (scale and sum as half2)
            slm_y_ds[blk] = *((const sycl::half2 *) &y[blk].ds);
        }
    }

    // Barrier: wait for Y-vector to be loaded to SLM
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Step 2: Each warp computes its row using Y from SLM
    if (row >= nrows) {
        return;
    }

    const block_q_t * x = (const block_q_t *) vx;

    // Hoist invariant iqs calculation outside the loop
    constexpr int qi_div_vdr      = qi / vdr;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int     base_iqs        = vdr * (lane_id % qi_div_vdr);
    const int     row_offset      = row * blocks_per_row;

    // 4-way accumulator for better ILP
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    // Main loop: 4x unrolled for ILP
    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        // Y block indices (for SLM lookup)
        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx0], slm_y_qs, slm_y_ds, iby0, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc1 += vec_dot_q_slm(&x[ibx1], slm_y_qs, slm_y_ds, iby1, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc2 += vec_dot_q_slm(&x[ibx2], slm_y_qs, slm_y_ds, iby2, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc3 += vec_dot_q_slm(&x[ibx3], slm_y_qs, slm_y_ds, iby3, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    // Handle remainder
    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx], slm_y_qs, slm_y_ds, iby, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    // Combine accumulators
    float tmp = (acc0 + acc1) + (acc2 + acc3);

    // Subgroup reduce
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

// Kernel name class for multi-row MMVQ
template <int qtype> class mmvq_multirow_kernel_name;

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void * __restrict__ vx,
                                       const void * __restrict__ vy,
                                       float * __restrict__ dst,
                                       const int                ncols,
                                       const int                nrows,
                                       const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int                ncols,
                                      const int                nrows,
                                      const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void * __restrict__ vx,
                                     const void * __restrict__ vy,
                                     float * __restrict__ dst,
                                     const int                ncols,
                                     const int                nrows,
                                     const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void * __restrict__ vx,
                                       const void * __restrict__ vy,
                                       float * __restrict__ dst,
                                       const int                ncols,
                                       const int                nrows,
                                       const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void * __restrict__ vx,
                                     const void * __restrict__ vy,
                                     float * __restrict__ dst,
                                     const int                ncols,
                                     const int                nrows,
                                     const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void * __restrict__ vx,
                                     const void * __restrict__ vy,
                                     float * __restrict__ dst,
                                     const int                ncols,
                                     const int                nrows,
                                     const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void * __restrict__ vx,
                                     const void * __restrict__ vy,
                                     float * __restrict__ dst,
                                     const int                ncols,
                                     const int                nrows,
                                     const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int                ncols,
                                      const int                nrows,
                                      const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void * __restrict__ vx,
                                      const void * __restrict__ vy,
                                      float * __restrict__ dst,
                                      const int                ncols,
                                      const int                nrows,
                                      const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);
    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        const int iqs =
            vdr * (item_ct1.get_local_id(2) % (qi / vdr));  // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

// Note: total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
//       row_low is the starting row offset for this slice (for split tensor support)
static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void *    vx,
                                               const void *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       total_nrows,
                                               const int       row_low,
                                               dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows, total_nrows,
                                                                              row_low, nd_item);
            });
    });
}

// Q8_0 reorder MMVQ dispatch function
// Note: total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
//       row_low is the starting row offset for this slice (for split tensor support)
static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void *    vx,
                                               const void *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       total_nrows,
                                               const int       row_low,
                                               dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows, total_nrows,
                                                                              row_low, nd_item);
            });
    });
}

// MXFP4 reorder MMVQ dispatch function
// Note: total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
//       row_low is the starting row offset for this slice (for split tensor support)
static void reorder_mul_mat_vec_mxfp4_q8_1_sycl(const void *    vx,
                                                const void *    vy,
                                                float *         dst,
                                                const int       ncols,
                                                const int       nrows,
                                                const int       total_nrows,
                                                const int       row_low,
                                                dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_MXFP4>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows, total_nrows,
                                                                               row_low, nd_item);
            });
    });
}

// GPU kernel to convert Q4_0 reordered format to warp-coalesced format
// Input layout (per tile):  [B0.qs[0:15]]...[B(TILE_BLOCKS-1).qs[0:15]] = block-major
// Output layout (per tile): [W0:B0..B(TILE_BLOCKS-1)]...[W3:B0..B(TILE_BLOCKS-1)] = word-major
// Where W0 = bytes 0-3, W1 = bytes 4-7, W2 = bytes 8-11, W3 = bytes 12-15 of each block
static void convert_q4_0_to_coalesced_kernel(const uint8_t * __restrict__ src,  // Reordered format
                                             uint8_t * __restrict__ dst,        // Coalesced format
                                             const int                ncols,
                                             const int                nrows,
                                             const sycl::nd_item<3> & item) {
    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK4_0;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;
    const int     bytes_per_row  = ncols / 2;  // 16 bytes per block, 32 elements per block

    // Grid: one work-item per 4-byte word in the tensor
    // Total words per row = blocks_per_row * 4 = tiles_per_row * 16 * 4
    const int global_id           = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 4;
    const int total_words         = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    // Decompose global_id into (row, word_in_row)
    const int row         = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    // Decompose word_in_row into (tile, block_in_tile, word_in_block)
    const int words_per_tile = TILE_BLOCKS * 4;  // 64 words per tile
    const int tile           = word_in_row / words_per_tile;
    const int word_in_tile   = word_in_row % words_per_tile;
    const int block_in_tile  = word_in_tile / 4;
    const int word_in_block  = word_in_tile % 4;

    // Source offset (block-major): row * bytes_per_row + tile * 256 + block * 16 + word * 4
    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + block_in_tile * 16 + word_in_block * 4;

    // Destination offset (word-major): row * bytes_per_row + tile * bytes + word * stride + block * 4
    const int dst_offset =
        row * bytes_per_row + tile * (TILE_BLOCKS * 16) + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

    // Copy 4 bytes
    *((int *) (dst + dst_offset)) = *((const int *) (src + src_offset));
}

// Convert Q4_0 reordered tensor to coalesced layout in-place
// Note: This modifies the tensor data directly. Only call once per tensor.
// WARNING: This must be called OUTSIDE of graph recording mode (cannot use wait() during recording)
static void convert_q4_0_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int blocks_per_row = ncols / QK4_0;
    const int total_words    = nrows * blocks_per_row * 4;  // 4 words per block

    // Allocate temporary buffer for conversion
    const int bytes_per_row = ncols / 2;
    const int total_bytes   = nrows * bytes_per_row;

    uint8_t * temp = ggml_sycl_malloc_device_tracked_t<uint8_t>(total_bytes, *stream, "mmvq_q4_coalesce");

    // Copy original quants to temp
    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    // Convert from temp to data (now coalesced)
    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        // Depend on copy completing before conversion
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_q4_0_to_coalesced_kernel(temp, (uint8_t *) data, ncols, nrows, item);
            });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task(
            [temp, total_bytes, stream]() { ggml_sycl_free_device_tracked_bytes(temp, total_bytes, *stream); });
    });
}

// Forward declarations for type-specific coalesced conversion functions
static void convert_q8_0_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream);
static void convert_mxfp4_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream);
static void convert_q6_k_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream);

// =============================================================================
// UNIFIED COALESCED REORDER FUNCTION
// This is the friend function declared in common.hpp - the ONLY way to set
// COALESCED mode. Atomically transforms data AND sets flag.
// Supports direct AoS→Coalesced or SoA→Coalesced, depending on current mode.
// =============================================================================
bool convert_tensor_to_coalesced(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller) {
    if (!tensor || !tensor->extra) {
        fprintf(stderr, "[COALESCED-UNIFIED] ERROR: tensor=%p extra=%p - cannot convert\n", (void *) tensor,
                tensor ? (void *) tensor->extra : nullptr);
        return false;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(tensor->extra);

    const reorder_mode current_mode = get_effective_reorder_mode(extra);

    // Check if already coalesced
    if (current_mode == reorder_mode::COALESCED) {
        return true;  // Already done
    }

    const int64_t ncols     = tensor->ne[0];
    const int64_t nrows     = ggml_nrows(tensor);
    int           device_id = -1;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));
    void * tensor_ptr = ggml_sycl_resolve_tensor_ptr(tensor, device_id);
    if (!tensor_ptr) {
        fprintf(stderr, "[COALESCED-UNIFIED] ERROR: tensor '%s' has no resolved device pointer\n",
                tensor->name ? tensor->name : "?");
        return false;
    }

    // Type-specific transform and tile alignment check
    int block_size = 0;
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            block_size = QK4_0;
            if ((ncols / block_size) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
                return false;  // Not tile-aligned
            }
            if (current_mode == reorder_mode::SOA) {
                convert_q4_0_to_coalesced_sycl(tensor_ptr, ncols, nrows, stream);
            } else if (current_mode == reorder_mode::NONE) {
                reorder_q4_0_aos_to_coalesced_sycl(tensor_ptr, tensor_ptr, ncols, nrows, stream);
            } else {
                return false;
            }
            break;

        case GGML_TYPE_Q8_0:
            block_size = QK8_0;
            if ((ncols / block_size) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
                return false;  // Not tile-aligned
            }
            if (current_mode == reorder_mode::SOA) {
                convert_q8_0_to_coalesced_sycl(tensor_ptr, ncols, nrows, stream);
            } else if (current_mode == reorder_mode::NONE) {
                reorder_q8_0_aos_to_coalesced_sycl(tensor_ptr, tensor_ptr, ncols, nrows, stream);
            } else {
                return false;
            }
            break;

        case GGML_TYPE_MXFP4:
            block_size = QK_MXFP4;
            if ((ncols / block_size) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
                return false;  // Not tile-aligned
            }
            if (current_mode == reorder_mode::SOA) {
                convert_mxfp4_to_coalesced_sycl(tensor_ptr, ncols, nrows, stream);
            } else if (current_mode == reorder_mode::NONE) {
                reorder_mxfp4_aos_to_coalesced_sycl(tensor_ptr, tensor_ptr, ncols, nrows, stream);
            } else {
                return false;
            }
            break;

        case GGML_TYPE_Q6_K:
            block_size = QK_K;
            if ((ncols / block_size) % MMVQ_COALESCED_TILE_BLOCKS != 0) {
                return false;  // Not tile-aligned
            }
            if (current_mode == reorder_mode::SOA) {
                convert_q6_k_to_coalesced_sycl(tensor_ptr, ncols, nrows, stream);
            } else if (current_mode == reorder_mode::NONE) {
                reorder_q6_k_aos_to_coalesced_sycl(tensor_ptr, tensor_ptr, ncols, nrows, stream);
            } else {
                return false;
            }
            break;

        default:
            fprintf(stderr, "[COALESCED-UNIFIED] ERROR: tensor '%s' type %d not supported\n",
                    tensor->name ? tensor->name : "?", tensor->type);
            return false;
    }

    // SET THE FLAG - mark tensor as COALESCED (via private friend access)
    extra->optimized_feature.set_reorder_mode_(reorder_mode::COALESCED, tensor->name, caller);

    // Keep unified layout descriptor in sync
    extra->layout.mode        = GGML_LAYOUT_COALESCED;
    extra->layout.data_ptr    = tensor_ptr;
    extra->layout.size        = ggml_nbytes(tensor);
    extra->layout.owns_memory = false;
    if (extra->layout.device_id < 0) {
        extra->layout.device_id = device_id;
    }
    extra->layout.qtype      = tensor->type;
    extra->layout.n_elements = ggml_nelements(tensor);
    extra->layout.n_experts  = tensor->ne[2] > 0 ? tensor->ne[2] : 1;

    GGML_SYCL_DEBUG("[COALESCED-UNIFIED] Converted '%s' (%lldx%lld) type=%d caller=%s\n",
                    tensor->name ? tensor->name : "?", (long long) ncols, (long long) nrows, tensor->type,
                    caller ? caller : "UNKNOWN");

    return true;
}

// Public API for coalesced conversion - call at model load time, after reorder
// This is a convenience wrapper that checks policy (env var, TP sharding) then
// calls the unified convert_tensor_to_coalesced() function.
bool ggml_sycl_convert_to_coalesced_q4_0(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Use centralized type support check
    if (!is_coalesced_supported(tensor->type)) {
        return false;
    }

    // Skip TP-sharded tensors (policy decision for this API)
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    if (extra && extra->tp_sharded) {
        return false;
    }

    // Use the unified function - it handles all validation and atomic transform+flag
    return convert_tensor_to_coalesced(tensor, stream, "ggml_sycl_convert_to_coalesced");
}

// Dispatch function for coalesced Q4_0 MMVQ kernel
static void coalesced_mul_mat_vec_q4_0_q8_1_sycl(const void *    vx,
                                                 const void *    vy,
                                                 float *         dst,
                                                 const int       ncols,
                                                 const int       nrows,
                                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    GGML_ASSERT((ncols / QK4_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q4_0_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// ============================================================================
// Q8_0 Warp-Coalesced MMVQ Kernel
// ============================================================================
// Q8_0 block: 32 int8 quants (32 bytes) + fp16 scale (2 bytes) = 34 bytes
// Coalesced layout: group consecutive words (4 bytes) across TILE_BLOCKS per tile
//
// Memory layout per tile (TILE_BLOCKS blocks):
// Source (block-major): [B0.W0-W7][B1.W0-W7]...[B(TILE_BLOCKS-1).W0-W7]
// Dest (word-major):    [W0:B0-B(TILE_BLOCKS-1)]...[W7:B0-B(TILE_BLOCKS-1)]
//
// Thread mapping: threads iterate block_in_tile and process both halves per block
static void mul_mat_vec_q8_0_coalesced(const void * __restrict__ vx,  // Coalesced X weights
                                       const void * __restrict__ vy,  // Reordered Y activations
                                       float * __restrict__ dst,
                                       const int                ncols,
                                       const int                nrows,
                                       const sycl::nd_item<3> & nd_item) {
    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  lane_id      = sg.get_local_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK8_0;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;

    const int word_stride = TILE_BLOCKS * 4;

    // X base pointers (coalesced layout: quants first, then scales)
    // Quants: tiles_per_row * TILE_BLOCKS * 32 bytes per row = ncols bytes
    const uint8_t * x_qs         = (const uint8_t *) vx;
    const int       x_row_stride = ncols;  // bytes per row of quants (32 bytes/block * blocks)

    // Scales are after all quants in the tensor
    const ggml_half * x_d = (const ggml_half *) ((const char *) vx + nrows * x_row_stride);

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t *      y_qs = (const int8_t *) vy;
    const sycl::half2 * y_ds = (const sycl::half2 *) ((const char *) vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_Q8_0;

        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            // Get scale for this block (scales are NOT coalesced, remain block-sequential)
            const int   block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
            const float d         = x_d[block_idx];

            // Y block index and base offset
            const int y_block = tile * TILE_BLOCKS + block_in_tile;
            const int y_base  = y_block * QK8_1;

            for (int half = 0; half < 2; ++half) {
                // Coalesced load: word w of block b at offset w*word_stride + b*4
                // Thread loads 4 words (16 bytes total) per half
                const int word_base = half * (4 * word_stride);

                // Perfectly coalesced 4-byte loads
                const int v0 = *((const int *) (x_qs + tile_base + word_base + 0 * word_stride + block_in_tile * 4));
                const int v1 = *((const int *) (x_qs + tile_base + word_base + 1 * word_stride + block_in_tile * 4));
                const int v2 = *((const int *) (x_qs + tile_base + word_base + 2 * word_stride + block_in_tile * 4));
                const int v3 = *((const int *) (x_qs + tile_base + word_base + 3 * word_stride + block_in_tile * 4));

                // Load Y data matching the X half we're processing
                // For half=0: Y elements 0-15
                // For half=1: Y elements 16-31
                const int y_offset = half * 16;                                                   // 0 or 16 (in bytes)
                const int u0       = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 0);  // Y[0:3] or Y[16:19]
                const int u1       = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);  // Y[4:7] or Y[20:23]
                const int u2       = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 2);  // Y[8:11] or Y[24:27]
                const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 3);  // Y[12:15] or Y[28:31]

                // dp4a: compute dot product of 4 int8 pairs
                int sumi = 0;
                sumi     = dpct::dp4a(v0, u0, sumi);
                sumi     = dpct::dp4a(v1, u1, sumi);
                sumi     = dpct::dp4a(v2, u2, sumi);
                sumi     = dpct::dp4a(v3, u3, sumi);

                // Apply scales: Q8_0 × Q8_1 = d8_0 * d8_1 * sumi
                const sycl::half2 ds8  = y_ds[y_block];
                const float       d8_1 = ds8[0];
                partial_sum += d * d8_1 * sumi;
            }
        }
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// GPU kernel to convert Q8_0 reordered format to warp-coalesced format
// Input layout (per tile):  [B0.qs[0:31]]...[B(TILE_BLOCKS-1).qs[0:31]] = block-major
// Output layout (per tile): [W0:B0..B(TILE_BLOCKS-1)]...[W7:B0..B(TILE_BLOCKS-1)] = word-major
// Where W0 = bytes 0-3, W1 = bytes 4-7, ..., W7 = bytes 28-31 of each block
static void convert_q8_0_to_coalesced_kernel(const uint8_t * __restrict__ src,  // Reordered format
                                             uint8_t * __restrict__ dst,        // Coalesced format
                                             const int                ncols,
                                             const int                nrows,
                                             const sycl::nd_item<3> & item) {
    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK8_0;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;
    const int     bytes_per_row  = ncols;  // 32 bytes per block, 32 elements per block

    // Grid: one work-item per 4-byte word in the tensor
    // Total words per row = blocks_per_row * 8 = tiles_per_row * TILE_BLOCKS * 8
    const int global_id           = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 8;  // 8 words per Q8_0 block
    const int total_words         = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    // Decompose global_id into (row, word_in_row)
    const int row         = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    // Decompose word_in_row into (tile, block_in_tile, word_in_block)
    const int words_per_tile = TILE_BLOCKS * 8;  // 128 words per tile
    const int tile           = word_in_row / words_per_tile;
    const int word_in_tile   = word_in_row % words_per_tile;
    const int block_in_tile  = word_in_tile / 8;
    const int word_in_block  = word_in_tile % 8;

    // Source offset (block-major): row * bytes_per_row + tile * 512 + block * 32 + word * 4
    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 32) + block_in_tile * 32 + word_in_block * 4;

    // Destination offset (word-major): row * bytes_per_row + tile * 512 + word * 64 + block * 4
    const int dst_offset =
        row * bytes_per_row + tile * (TILE_BLOCKS * 32) + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

    // Copy 4 bytes
    *((int *) (dst + dst_offset)) = *((const int *) (src + src_offset));
}

// Convert Q8_0 reordered tensor to coalesced layout in-place
static void convert_q8_0_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int blocks_per_row = ncols / QK8_0;
    const int total_words    = nrows * blocks_per_row * 8;  // 8 words per block

    // Allocate temporary buffer for conversion
    const int bytes_per_row = ncols;
    const int total_bytes   = nrows * bytes_per_row;

    uint8_t * temp = ggml_sycl_malloc_device_tracked_t<uint8_t>(total_bytes, *stream, "mmvq_temp");

    // Copy original quants to temp
    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    // Convert from temp to data (now coalesced)
    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_q8_0_to_coalesced_kernel(temp, (uint8_t *) data, ncols, nrows, item);
            });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task(
            [temp, total_bytes, stream]() { ggml_sycl_free_device_tracked_bytes(temp, total_bytes, *stream); });
    });
}

// Public API for Q8_0 coalesced conversion - call at model load time, after reorder
bool ggml_sycl_convert_to_coalesced_q8_0(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Use centralized type support check (Q8_0 not yet enabled)
    if (!is_coalesced_supported(tensor->type)) {
        return false;
    }

    // Skip TP-sharded tensors (policy decision for this API)
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    if (extra && extra->tp_sharded) {
        return false;
    }

    // Use the unified function - it handles all validation and atomic transform+flag
    return convert_tensor_to_coalesced(tensor, stream, "ggml_sycl_convert_to_coalesced");
}

// Public API for Q6_K coalesced conversion - call at model load time, after reorder
bool ggml_sycl_convert_to_coalesced_q6_k(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Use centralized type support check
    if (!is_coalesced_supported(tensor->type)) {
        return false;
    }

    // Skip TP-sharded tensors (policy decision for this API)
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    if (extra && extra->tp_sharded) {
        return false;
    }

    // Use the unified function - it handles all validation and atomic transform+flag
    return convert_tensor_to_coalesced(tensor, stream, "ggml_sycl_convert_to_coalesced");
}

// Dispatch function for coalesced Q8_0 MMVQ kernel
static void coalesced_mul_mat_vec_q8_0_q8_1_sycl(const void *    vx,
                                                 const void *    vy,
                                                 float *         dst,
                                                 const int       ncols,
                                                 const int       nrows,
                                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    GGML_ASSERT((ncols / QK8_0) % MMVQ_COALESCED_TILE_BLOCKS == 0);  // Must be multiple of tile size

    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q8_0_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// ============================================================================
// MXFP4 Warp-Coalesced MMVQ Kernel
// ============================================================================
// MXFP4 block: 16 packed bytes (32 4-bit elements) + 1 byte E8M0 exponent = 17 bytes
// Same coalesced layout as Q4_0 (16 bytes quants per block)
static void mul_mat_vec_mxfp4_coalesced(const void * __restrict__ vx,  // Coalesced X weights
                                        const void * __restrict__ vy,  // Reordered Y activations
                                        float * __restrict__ dst,
                                        const int                ncols,
                                        const int                nrows,
                                        const sycl::nd_item<3> & nd_item) {
    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  lane_id      = sg.get_local_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK_MXFP4;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;

    // X base pointers (coalesced layout: quants first, then scales)
    const uint8_t * x_qs         = (const uint8_t *) vx;
    const int       x_row_stride = ncols / 2;  // bytes per row of quants (16 bytes/block)

    // Scales are after all quants in the tensor (1 byte E8M0 per block)
    const uint8_t * x_e = (const uint8_t *) vx + nrows * x_row_stride;

    // Y base pointers (standard reordered format: quants, then ds)
    const int8_t *      y_qs = (const int8_t *) vy;
    const sycl::half2 * y_ds = (const sycl::half2 *) ((const char *) vy + ncols);

    float partial_sum = 0.0f;

    for (int tile = 0; tile < tiles_per_row; tile++) {
        // Base offset for this tile's quants
        const int tile_base = row * x_row_stride + tile * MMVQ_COALESCED_TILE_BYTES_MXFP4;

        const int word_stride = TILE_BLOCKS * 4;
        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            for (int half = 0; half < 2; ++half) {
                const int word_base    = half * (2 * word_stride);
                const int word0_offset = word_base + block_in_tile * 4;                // word 0 or 2
                const int word1_offset = word_base + word_stride + block_in_tile * 4;  // word 1 or 3

                // Perfectly coalesced 4-byte loads
                const int v0 = *((const int *) (x_qs + tile_base + word0_offset));
                const int v1 = *((const int *) (x_qs + tile_base + word1_offset));

                // Get E8M0 exponent for this block
                const int     block_idx = row * blocks_per_row + tile * TILE_BLOCKS + block_in_tile;
                const uint8_t e8m0      = x_e[block_idx];
                const float   scale     = ggml_sycl_e8m0_to_fp32(e8m0) * 0.5f;

                // Y block index and base offset
                const int y_block = tile * TILE_BLOCKS + block_in_tile;
                const int y_base  = y_block * QK8_1;

                // Load Y data matching the X half we're processing
                const int y_offset = half * 8;  // 0 or 8 (in terms of bytes)

                // Use MXFP4 lookup table for dequantization
                // Process 8 elements per load (2 words of 4 nibbles each = 8 FP4 values)
                const sycl::int2 dq0 = get_int_from_table_16(v0, kvalues_mxfp4);
                const sycl::int2 dq1 = get_int_from_table_16(v1, kvalues_mxfp4);

                // Load corresponding Y values
                const int u0 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4);      // Y[0:3] or Y[8:11]
                const int u1 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 4);  // Y[16:19] or Y[24:27]
                const int u2 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 1);  // Y[4:7] or Y[12:15]
                const int u3 = get_int_from_int8_aligned(y_qs + y_base, y_offset / 4 + 5);  // Y[20:23] or Y[28:31]

                // dp4a: compute dot product
                int sumi = 0;
                sumi     = ggml_sycl_dp4a(dq0.x(), u0, sumi);
                sumi     = ggml_sycl_dp4a(dq0.y(), u1, sumi);
                sumi     = ggml_sycl_dp4a(dq1.x(), u2, sumi);
                sumi     = ggml_sycl_dp4a(dq1.y(), u3, sumi);

                // Apply scales
                const sycl::half2 ds8  = y_ds[y_block];
                const float       d8_1 = ds8[0];
                partial_sum += scale * d8_1 * sumi;
            }
        }
    }

    // Warp reduction using subgroup intrinsic
    auto sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

// GPU kernel to convert MXFP4 reordered format to warp-coalesced format
// Same layout as Q4_0 (16 bytes quants per block)
static void convert_mxfp4_to_coalesced_kernel(const uint8_t * __restrict__ src,
                                              uint8_t * __restrict__ dst,
                                              const int                ncols,
                                              const int                nrows,
                                              const sycl::nd_item<3> & item) {
    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK_MXFP4;
    const int     bytes_per_row  = ncols / 2;  // 16 bytes per block, 32 elements per block

    const int global_id           = item.get_global_linear_id();
    const int total_words_per_row = blocks_per_row * 4;  // 4 words per MXFP4 block
    const int total_words         = nrows * total_words_per_row;

    if (global_id >= total_words) {
        return;
    }

    const int row         = global_id / total_words_per_row;
    const int word_in_row = global_id % total_words_per_row;

    const int words_per_tile = TILE_BLOCKS * 4;
    const int tile           = word_in_row / words_per_tile;
    const int word_in_tile   = word_in_row % words_per_tile;
    const int block_in_tile  = word_in_tile / 4;
    const int word_in_block  = word_in_tile % 4;

    const int src_offset = row * bytes_per_row + tile * (TILE_BLOCKS * 16) + block_in_tile * 16 + word_in_block * 4;
    const int dst_offset =
        row * bytes_per_row + tile * (TILE_BLOCKS * 16) + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

    *((int *) (dst + dst_offset)) = *((const int *) (src + src_offset));
}

static void convert_mxfp4_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_ASSERT((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int blocks_per_row = ncols / QK_MXFP4;
    const int total_words    = nrows * blocks_per_row * 4;

    const int bytes_per_row = ncols / 2;
    const int total_bytes   = nrows * bytes_per_row;

    uint8_t * temp = ggml_sycl_malloc_device_tracked_t<uint8_t>(total_bytes, *stream, "mmvq_temp");

    sycl::event copy_event = stream->memcpy(temp, data, total_bytes);

    const int block_size = 256;
    const int num_blocks = (total_words + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_mxfp4_to_coalesced_kernel(temp, (uint8_t *) data, ncols, nrows, item);
            });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task(
            [temp, total_bytes, stream]() { ggml_sycl_free_device_tracked_bytes(temp, total_bytes, *stream); });
    });
}

// Q6_K Coalesced Conversion
// SoA layout: [all ql][all qh][all scales][all d]
// Coalesced layout per tile (TILE_BLOCKS blocks):
//   [ql word-major: TILE_BLOCKS * (QK_K / 2)]
//   [qh word-major: TILE_BLOCKS * (QK_K / 4)]
//   [scales word-major: TILE_BLOCKS * (QK_K / 16)]
// D values remain at tensor end (not coalesced within tiles)
static void convert_q6_k_to_coalesced_kernel(const uint8_t * __restrict__ src,  // SoA format
                                             uint8_t * __restrict__ dst,        // Coalesced format
                                             const int                ncols,
                                             const int                nrows,
                                             const sycl::nd_item<3> & item) {
    constexpr int TILE_BLOCKS    = MMVQ_COALESCED_TILE_BLOCKS;
    const int     blocks_per_row = ncols / QK_K;
    const int     tiles_per_row  = blocks_per_row / TILE_BLOCKS;

    // SoA region sizes per row
    const int ql_per_row = blocks_per_row * (QK_K / 2);   // 128 bytes/block
    const int qh_per_row = blocks_per_row * (QK_K / 4);   // 64 bytes/block
    const int sc_per_row = blocks_per_row * (QK_K / 16);  // 16 bytes/block

    // SoA base pointers
    const uint8_t * src_ql = src;
    const uint8_t * src_qh = src_ql + nrows * ql_per_row;
    const uint8_t * src_sc = src_qh + nrows * qh_per_row;
    // D values stay in place (at tensor end)

    // Coalesced tile sizes
    constexpr int ql_tile_bytes = TILE_BLOCKS * (QK_K / 2);
    constexpr int qh_tile_bytes = TILE_BLOCKS * (QK_K / 4);
    constexpr int sc_tile_bytes = TILE_BLOCKS * (QK_K / 16);
    constexpr int tile_total    = ql_tile_bytes + qh_tile_bytes + sc_tile_bytes;

    // Destination base for quant data (coalesced tiles)
    uint8_t * dst_tiles = dst;
    // D values remain at same location as SoA (after all ql+qh+sc)

    // Each thread handles one word (4 bytes) reordering
    const int global_id     = item.get_global_linear_id();
    const int total_threads = item.get_global_range().size();

    // Process ql words (32 words per block, 512 words per tile)
    const int ql_words_per_tile = TILE_BLOCKS * 32;
    const int total_ql_words    = nrows * tiles_per_row * ql_words_per_tile;

    for (int i = global_id; i < total_ql_words; i += total_threads) {
        const int tile_word = i % ql_words_per_tile;
        const int tile_idx  = (i / ql_words_per_tile) % tiles_per_row;
        const int row       = i / (tiles_per_row * ql_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        // Source: SoA sequential blocks
        const int src_offset =
            row * ql_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 2) + word_in_block * 4;

        // Dest: word-major within tile
        const int dst_tile_base = row * tiles_per_row * tile_total + tile_idx * tile_total;
        const int dst_offset    = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *) (dst_tiles + dst_offset)) = *((const int *) (src_ql + src_offset));
    }

    // Process qh words (16 words per block, 256 words per tile)
    const int qh_words_per_tile = TILE_BLOCKS * 16;
    const int total_qh_words    = nrows * tiles_per_row * qh_words_per_tile;

    for (int i = global_id; i < total_qh_words; i += total_threads) {
        const int tile_word = i % qh_words_per_tile;
        const int tile_idx  = (i / qh_words_per_tile) % tiles_per_row;
        const int row       = i / (tiles_per_row * qh_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        const int src_offset =
            row * qh_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 4) + word_in_block * 4;
        const int dst_tile_base = row * tiles_per_row * tile_total + tile_idx * tile_total + ql_tile_bytes;
        const int dst_offset    = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *) (dst_tiles + dst_offset)) = *((const int *) (src_qh + src_offset));
    }

    // Process scales words (4 words per block, 64 words per tile)
    const int sc_words_per_tile = TILE_BLOCKS * 4;
    const int total_sc_words    = nrows * tiles_per_row * sc_words_per_tile;

    for (int i = global_id; i < total_sc_words; i += total_threads) {
        const int tile_word = i % sc_words_per_tile;
        const int tile_idx  = (i / sc_words_per_tile) % tiles_per_row;
        const int row       = i / (tiles_per_row * sc_words_per_tile);

        const int word_in_block = tile_word / TILE_BLOCKS;
        const int block_in_tile = tile_word % TILE_BLOCKS;

        const int src_offset =
            row * sc_per_row + (tile_idx * TILE_BLOCKS + block_in_tile) * (QK_K / 16) + word_in_block * 4;
        const int dst_tile_base =
            row * tiles_per_row * tile_total + tile_idx * tile_total + ql_tile_bytes + qh_tile_bytes;
        const int dst_offset = dst_tile_base + word_in_block * (TILE_BLOCKS * 4) + block_in_tile * 4;

        *((int *) (dst_tiles + dst_offset)) = *((const int *) (src_sc + src_offset));
    }
}

static void convert_q6_k_to_coalesced_sycl(void * data, const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    GGML_ASSERT((ncols / QK_K) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int blocks_per_row = ncols / QK_K;
    const int tiles_per_row  = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;

    // Total quant bytes (ql + qh + scales, excluding d)
    const size_t quant_bytes_per_row = blocks_per_row * ((QK_K / 2) + (QK_K / 4) + (QK_K / 16));
    const size_t total_quant_bytes   = nrows * quant_bytes_per_row;

    // Allocate temp buffer for in-place conversion
    const int runtime_device = ggml_sycl_get_device_id_from_queue(*stream);
    uint8_t * temp           = (uint8_t *) ggml_sycl_malloc_device(total_quant_bytes, *stream, "mmvq_quant_temp");
    GGML_ASSERT(temp != nullptr);

    // Copy current data to temp
    sycl::event copy_event = stream->memcpy(temp, data, total_quant_bytes);

    // Launch conversion kernel
    const int total_work = nrows * tiles_per_row * MMVQ_COALESCED_TILE_BLOCKS * 32;  // ql dominant
    const int block_size = 256;
    const int num_blocks = (total_work + block_size - 1) / block_size;

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks * block_size), sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item) {
                convert_q6_k_to_coalesced_kernel(temp, (uint8_t *) data, ncols, nrows, item);
            });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(convert_event);
        cgh.host_task([temp, stream, total_quant_bytes]() {
            ggml_sycl_free_device_tracked_bytes(temp, total_quant_bytes, *stream);
        });
    });
}

// Public API for MXFP4 coalesced conversion
bool ggml_sycl_convert_to_coalesced_mxfp4(const ggml_tensor * tensor, dpct::queue_ptr stream) {
    // Use centralized type support check (MXFP4 not yet enabled)
    if (!is_coalesced_supported(tensor->type)) {
        return false;
    }

    // Skip TP-sharded tensors (policy decision for this API)
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;
    if (extra && extra->tp_sharded) {
        return false;
    }

    // Use the unified function - it handles all validation and atomic transform+flag
    return convert_tensor_to_coalesced(tensor, stream, "ggml_sycl_convert_to_coalesced");
}

// Dispatch function for coalesced MXFP4 MMVQ kernel
static void coalesced_mul_mat_vec_mxfp4_q8_1_sycl(const void *    vx,
                                                  const void *    vy,
                                                  float *         dst,
                                                  const int       ncols,
                                                  const int       nrows,
                                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_ASSERT((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_coalesced_kernel_name<GGML_TYPE_MXFP4>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_mxfp4_coalesced(vx, vy, dst, ncols, nrows, nd_item);
            });
    });
}

// Public wrapper for coalesced MXFP4 MMVQ kernel (used by direct dispatch in ggml-sycl.cpp)
void mmvq_submit_mxfp4_coalesced(sycl::queue & q, const void * vx, const void * vy, float * dst, int ncols, int nrows) {
    coalesced_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, &q);
}

// Dispatch function for coalesced Q6_K MMVQ kernel
// Now uses variable tile kernel which handles any block count
// total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
// row_low is the starting row offset for this slice
static void coalesced_mul_mat_vec_q6_k_q8_1_sycl(const void *    vx,
                                                 const void *    vy,
                                                 float *         dst,
                                                 const int       ncols,
                                                 const int       nrows,
                                                 const int       total_nrows,
                                                 const int       row_low,
                                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Use variable tile kernel which handles arbitrary block counts
    variable_tile_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, total_nrows, row_low, *stream);
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);

    // Use multi-row kernel with SLM Y-vector sharing for better bandwidth utilization
    constexpr int        NROWS_PER_WG = MMVQ_NROWS_PER_WG;
    const int            block_num_z  = (nrows + NROWS_PER_WG - 1) / NROWS_PER_WG;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, NROWS_PER_WG, WARP_SIZE);

    const int blocks_per_row = ncols / QK4_0;
    const int slm_y_qs_size  = blocks_per_row * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size  = blocks_per_row + 1;  // +1 for padding

    stream->submit([&](sycl::handler & cgh) {
        // Allocate SLM for Y-vector (shared across all rows in work-group)
        sycl::local_accessor<int, 1>         slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_multirow_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_multirow<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1_slm,
                                       NROWS_PER_WG>(vx, vy, dst, ncols, nrows, item_ct1, slm_y_qs.get_pointer(),
                                                     slm_y_ds.get_pointer());
            });
    });
}

// MoE dispatch: Q4_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q4_0_q8_1_id_sycl(const void *         vx,
                                          const void * const * expert_ptrs,
                                          const void *         vy,
                                          float *              dst,
                                          const int32_t *      ids,
                                          const int            ncols,
                                          const int            nrows_per_expert,
                                          const int            total_batches,
                                          const int            n_ids,
                                          const int            n_tokens,
                                          const int            ne11,
                                          const int64_t        stride_expert_x,
                                          const int64_t        ids_nb0,
                                          const int64_t        ids_nb1,
                                          const int64_t        nb11,
                                          const int64_t        nb12,
                                          const int64_t        nb1,
                                          const int64_t        nb2,
                                          dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr int moe_mmv_y = GGML_SYCL_MOE_MMV_Y;
    static_assert(moe_mmv_y * WARP_SIZE <= 1024, "GGML_SYCL_MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + moe_mmv_y - 1) / moe_mmv_y;
    // 3D dispatch: (1, total_batches, block_num_z) - batch in y dimension, rows in z
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, moe_mmv_y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_id_kernel_name<GGML_TYPE_Q4_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_id<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                    vx, expert_ptrs, vy, dst, ids, ncols, nrows_per_expert, n_ids, n_tokens, ne11, stride_expert_x,
                    ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
            });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q4_1>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(vx, vy, dst, ncols,
                                                                                                   nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void *    vx,
                                        const void *    vy,
                                        float *         dst,
                                        const int       ncols,
                                        const int       nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_MXFP4>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_q5_0_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_0>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(vx, vy, dst, ncols,
                                                                                                   nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_1>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(vx, vy, dst, ncols,
                                                                                                   nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    // Use multi-row kernel with SLM Y-vector sharing for better bandwidth utilization
    constexpr int        NROWS_PER_WG = MMVQ_NROWS_PER_WG;
    const int            block_num_z  = (nrows + NROWS_PER_WG - 1) / NROWS_PER_WG;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, NROWS_PER_WG, WARP_SIZE);

    const int blocks_per_row = ncols / QK8_0;
    const int slm_y_qs_size  = blocks_per_row * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size  = blocks_per_row + 1;  // +1 for padding

    stream->submit([&](sycl::handler & cgh) {
        // Allocate SLM for Y-vector (shared across all rows in work-group)
        sycl::local_accessor<int, 1>         slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_multirow_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_multirow<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1_slm,
                                       NROWS_PER_WG>(vx, vy, dst, ncols, nrows, item_ct1, slm_y_qs.get_pointer(),
                                                     slm_y_ds.get_pointer());
            });
    });
}

// MoE dispatch: Q8_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q8_0_q8_1_id_sycl(const void *         vx,
                                          const void * const * expert_ptrs,
                                          const void *         vy,
                                          float *              dst,
                                          const int32_t *      ids,
                                          const int            ncols,
                                          const int            nrows_per_expert,
                                          const int            total_batches,
                                          const int            n_ids,
                                          const int            n_tokens,
                                          const int            ne11,
                                          const int64_t        stride_expert_x,
                                          const int64_t        ids_nb0,
                                          const int64_t        ids_nb1,
                                          const int64_t        nb11,
                                          const int64_t        nb12,
                                          const int64_t        nb1,
                                          const int64_t        nb2,
                                          dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr int moe_mmv_y = GGML_SYCL_MOE_MMV_Y;
    static_assert(moe_mmv_y * WARP_SIZE <= 1024, "GGML_SYCL_MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + moe_mmv_y - 1) / moe_mmv_y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, moe_mmv_y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_id_kernel_name<GGML_TYPE_Q8_0>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_id<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                    vx, expert_ptrs, vy, dst, ids, ncols, nrows_per_expert, n_ids, n_tokens, ne11, stride_expert_x,
                    ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
            });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q2_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(vx, vy, dst, ncols,
                                                                                                  nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q3_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(vx, vy, dst, ncols,
                                                                                                  nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q4_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(vx, vy, dst, ncols,
                                                                                                  nrows, item_ct1);
                });
        });
    }
}

// Note: total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
//       row_low is the starting row offset for this slice (for split tensor support)
static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void *    vx,
                                               const void *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       total_nrows,
                                               const int       row_low,
                                               dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q4_K>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols, nrows, total_nrows,
                                                                              row_low, nd_item);
            });
    });
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q5_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols,
                                                                                                  nrows, item_ct1);
                });
        });
    }
}

// Note: total_nrows is the full tensor row count (ne01), nrows is the slice size (row_diff)
//       row_low is the starting row offset for this slice (for split tensor support)
static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void *    vx,
                                               const void *    vy,
                                               float *         dst,
                                               const int       ncols,
                                               const int       nrows,
                                               const int       total_nrows,
                                               const int       row_low,
                                               dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_reorder_kernel_name<GGML_TYPE_Q6_K>>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows, total_nrows,
                                                                              row_low, nd_item);
            });
    });
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *    vx,
                                       const void *    vy,
                                       float *         dst,
                                       const int       ncols,
                                       const int       nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_Q6_K>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(vx, vy, dst, ncols,
                                                                                                  nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *    vx,
                                          const void *    vy,
                                          float *         dst,
                                          const int       ncols,
                                          const int       nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS / 2, block_iq2_xxs, 1>(vx, vy, dst, ncols,
                                                                                                 nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *    vx,
                                         const void *    vy,
                                         float *         dst,
                                         const int       ncols,
                                         const int       nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS / 2, block_iq2_xs, 1>(vx, vy, dst, ncols, nrows,
                                                                                              item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *    vx,
                                        const void *    vy,
                                        float *         dst,
                                        const int       ncols,
                                        const int       nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S / 2, block_iq2_s, 1>(vx, vy, dst, ncols, nrows,
                                                                                           item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *    vx,
                                          const void *    vy,
                                          float *         dst,
                                          const int       ncols,
                                          const int       nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS / 2, block_iq3_xxs, 1>(vx, vy, dst, ncols,
                                                                                                 nrows, item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *    vx,
                                        const void *    vy,
                                        float *         dst,
                                        const int       ncols,
                                        const int       nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S / 2, block_iq3_s, 1>(vx, vy, dst, ncols, nrows,
                                                                                           item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *    vx,
                                        const void *    vy,
                                        float *         dst,
                                        const int       ncols,
                                        const int       nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(vx, vy, dst, ncols, nrows,
                                                                                       item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *    vx,
                                        const void *    vy,
                                        float *         dst,
                                        const int       ncols,
                                        const int       nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(vx, vy, dst, ncols, nrows,
                                                                                       item_ct1);
                             });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *    vx,
                                         const void *    vy,
                                         float *         dst,
                                         const int       ncols,
                                         const int       nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for<mmvq_kernel_name<GGML_TYPE_IQ4_NL>>(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(vx, vy, dst, ncols, nrows, item_ct1);
                });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *    vx,
                                         const void *    vy,
                                         float *         dst,
                                         const int       ncols,
                                         const int       nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS / 4, block_iq4_xs, 1>(vx, vy, dst, ncols, nrows,
                                                                                              item_ct1);
                             });
        });
    }
}

// MoE-aware MXFP4 kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_mxfp4_q8_1_id(const void * __restrict__ vx,
                                        const void * __restrict__ vy,
                                        float * __restrict__ dst,
                                        const int32_t * __restrict__ ids,
                                        const int                ncols,
                                        const int                nrows_per_expert,
                                        const int                n_ids,
                                        const int                n_tokens,
                                        const int                ne11,
                                        const int64_t            stride_expert_x,
                                        const int64_t            ids_nb0,
                                        const int64_t            ids_nb1,
                                        const int64_t            nb11,
                                        const int64_t            nb12,
                                        const int64_t            nb1,
                                        const int64_t            nb2,
                                        const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - row-major by id
    const int id   = batch_idx / n_tokens;       // Expert selection index
    const int iid1 = batch_idx - id * n_tokens;  // Token position

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1  = id;
    const int64_t i2  = iid1;

    const int blocks_per_row  = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    float tmp = 0.0f;

    // Expert weights: offset by expert_id * stride_expert_x
    const block_q_t *  x = (const block_q_t *) ((const char *) vx + expert_id * stride_expert_x);
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char *) vy + i11 * nb11 + i12 * nb12);

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);
        const int iqs = vdr * (item_ct1.get_local_id(2) % (qi / vdr));

        tmp += vec_dot_mxfp4_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float *) ((char *) dst + i1 * nb1 + i2 * nb2);
        dst_out[row]    = tmp;
    }
}

// MoE dispatch: MXFP4 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_mxfp4_q8_1_id_sycl(const void *         vx,
                                           const void * const * expert_ptrs,
                                           const void *         vy,
                                           float *              dst,
                                           const int32_t *      ids,
                                           const int            ncols,
                                           const int            nrows_per_expert,
                                           const int            total_batches,
                                           const int            n_ids,
                                           const int            n_tokens,
                                           const int            ne11,
                                           const int64_t        stride_expert_x,
                                           const int64_t        ids_nb0,
                                           const int64_t        ids_nb1,
                                           const int64_t        nb11,
                                           const int64_t        nb12,
                                           const int64_t        nb1,
                                           const int64_t        nb2,
                                           dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr int moe_mmv_y = GGML_SYCL_MOE_MMV_Y;
    static_assert(moe_mmv_y * WARP_SIZE <= 1024, "GGML_SYCL_MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + moe_mmv_y - 1) / moe_mmv_y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, moe_mmv_y, WARP_SIZE);

    // Use generic template with vec_dot_mxfp4_q8_1 function pointer
    // This matches how Q4_0 and Q8_0 work
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                 vx, expert_ptrs, vy, dst, ids, ncols, nrows_per_expert, n_ids, n_tokens, ne11,
                                 stride_expert_x, ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

template <bool USE_WEIGHT_SCALE = true>
static __dpct_inline__ float mxfp4_soa_q8_1_block_dot(const uint8_t * __restrict__ qs_row,
                                                      const uint8_t * __restrict__ scale_row,
                                                      const char * __restrict__ y_base,
                                                      const int ncols_y,
                                                      const int b) {
    const int *         q8_qs = (const int *) (y_base + b * QK8_1);
    const sycl::half2 * q8_ds = (const sycl::half2 *) (y_base + ncols_y + b * sizeof(sycl::half2));
    const float         d8    = (float) (*q8_ds)[0];

    const uint8_t * qs = qs_row + b * (QK_MXFP4 / 2);

    int sumi = 0;
#pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
        const int        aux_q4 = *((const int *) (qs + i));
        const sycl::int2 v      = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_sycl_dp4a(v.x(), q8_qs[i / 4], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8_qs[i / 4 + 4], sumi);
    }

    if constexpr (USE_WEIGHT_SCALE) {
        const uint8_t e8m0 = scale_row[b];
        const float   d    = ggml_sycl_e8m0_to_fp32(e8m0) * 0.5f;
        return d * d8 * sumi;
    } else {
        GGML_UNUSED(scale_row);
        return d8 * sumi;
    }
}

template <bool USE_WEIGHT_SCALE = true>
static __dpct_inline__ void mxfp4_soa_q8_1_block_dot_pair(const uint8_t * __restrict__ qs_row_a,
                                                          const uint8_t * __restrict__ scale_row_a,
                                                          const uint8_t * __restrict__ qs_row_b,
                                                          const uint8_t * __restrict__ scale_row_b,
                                                          const char * __restrict__ y_base,
                                                          const int ncols_y,
                                                          const int b,
                                                          float &   acc_a,
                                                          float &   acc_b) {
    const int *         q8_qs = (const int *) (y_base + b * QK8_1);
    const sycl::half2 * q8_ds = (const sycl::half2 *) (y_base + ncols_y + b * sizeof(sycl::half2));
    const float         d8    = (float) (*q8_ds)[0];

    const uint8_t * qs_a = qs_row_a + b * (QK_MXFP4 / 2);
    const uint8_t * qs_b = qs_row_b + b * (QK_MXFP4 / 2);

    int sumi_a = 0;
    int sumi_b = 0;
#pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
        const int aux_q4_a = *((const int *) (qs_a + i));
        const int aux_q4_b = *((const int *) (qs_b + i));

        const sycl::int2 va = get_int_from_table_16(aux_q4_a, kvalues_mxfp4);
        const sycl::int2 vb = get_int_from_table_16(aux_q4_b, kvalues_mxfp4);

        const int q8_0 = q8_qs[i / 4];
        const int q8_1 = q8_qs[i / 4 + 4];

        sumi_a = ggml_sycl_dp4a(va.x(), q8_0, sumi_a);
        sumi_a = ggml_sycl_dp4a(va.y(), q8_1, sumi_a);
        sumi_b = ggml_sycl_dp4a(vb.x(), q8_0, sumi_b);
        sumi_b = ggml_sycl_dp4a(vb.y(), q8_1, sumi_b);
    }

    if constexpr (USE_WEIGHT_SCALE) {
        const float d_a = ggml_sycl_e8m0_to_fp32(scale_row_a[b]) * 0.5f;
        const float d_b = ggml_sycl_e8m0_to_fp32(scale_row_b[b]) * 0.5f;
        acc_a += d_a * d8 * sumi_a;
        acc_b += d_b * d8 * sumi_b;
    } else {
        GGML_UNUSED(scale_row_a);
        GGML_UNUSED(scale_row_b);
        acc_a += d8 * sumi_a;
        acc_b += d8 * sumi_b;
    }
}

template <bool USE_WEIGHT_SCALE = true>
static __dpct_inline__ void mxfp4_soa_q8_1_block_dot_pair_slm(const uint8_t * __restrict__ qs_row_a,
                                                              const uint8_t * __restrict__ scale_row_a,
                                                              const uint8_t * __restrict__ qs_row_b,
                                                              const uint8_t * __restrict__ scale_row_b,
                                                              const int * __restrict__ y_qs,
                                                              const sycl::half2 * __restrict__ y_ds,
                                                              const int y_qs_stride,
                                                              const int b,
                                                              float &   acc_a,
                                                              float &   acc_b) {
    const int *         q8_qs = y_qs + b * y_qs_stride;
    const sycl::half2 * q8_ds = y_ds + b;
    const float         d8    = (float) (*q8_ds)[0];

    const uint8_t * qs_a = qs_row_a + b * (QK_MXFP4 / 2);
    const uint8_t * qs_b = qs_row_b + b * (QK_MXFP4 / 2);

    int sumi_a = 0;
    int sumi_b = 0;
#pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; i += 4) {
        const int aux_q4_a = *((const int *) (qs_a + i));
        const int aux_q4_b = *((const int *) (qs_b + i));

        const sycl::int2 va = get_int_from_table_16(aux_q4_a, kvalues_mxfp4);
        const sycl::int2 vb = get_int_from_table_16(aux_q4_b, kvalues_mxfp4);

        const int q8_0 = q8_qs[i / 4];
        const int q8_1 = q8_qs[i / 4 + 4];

        sumi_a = ggml_sycl_dp4a(va.x(), q8_0, sumi_a);
        sumi_a = ggml_sycl_dp4a(va.y(), q8_1, sumi_a);
        sumi_b = ggml_sycl_dp4a(vb.x(), q8_0, sumi_b);
        sumi_b = ggml_sycl_dp4a(vb.y(), q8_1, sumi_b);
    }

    if constexpr (USE_WEIGHT_SCALE) {
        const float d_a = ggml_sycl_e8m0_to_fp32(scale_row_a[b]) * 0.5f;
        const float d_b = ggml_sycl_e8m0_to_fp32(scale_row_b[b]) * 0.5f;
        acc_a += d_a * d8 * sumi_a;
        acc_b += d_b * d8 * sumi_b;
    } else {
        GGML_UNUSED(scale_row_a);
        GGML_UNUSED(scale_row_b);
        acc_a += d8 * sumi_a;
        acc_b += d8 * sumi_b;
    }
}

// MoE dispatch: MXFP4 with SoA layout (reordered weights) + expert routing
// SoA layout for weights: [all qs for all experts][all scales for all experts]
// SoA layout for Q8_1: per row [quants: ncols_y bytes][ds: nblocks * sizeof(half2)]
// After reorder_qw_mxfp4: qs at offset 0, scales at offset (ncols/2)*total_rows
template <bool USE_WEIGHT_SCALE = true>
static void mul_mat_vec_mxfp4_q8_1_soa_id_kernel(
    const uint8_t * __restrict__ vx,                   // Base pointer to reordered MXFP4 tensor (SoA layout)
    const uint8_t * const * __restrict__ expert_ptrs,  // Optional per-expert base pointers (SoA layout)
    const void * __restrict__ vy,                      // Base pointer to Q8_1 tensor (SoA layout)
    float * __restrict__ dst,
    const int32_t * __restrict__ ids,
    const int                ncols,    // MXFP4 tensor columns
    const int                ncols_y,  // Q8_1 row size (for SoA ds offset)
    const int                nrows_per_expert,
    const int                n_ids,
    const int                n_tokens,
    const int                ne11,
    const int64_t            total_qs_size,             // Offset to MXFP4 scale region
    const int64_t            total_qs_size_per_expert,  // Offset to MXFP4 scale region (per-expert layout)
    const int64_t            ids_nb0,
    const int64_t            ids_nb1,
    const int64_t            nb11,
    const int64_t            nb12,
    const int64_t            nb1,
    const int64_t            nb2,
    const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - row-major by id
    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    int32_t expert_id = -1;
    if (ids) {
        // Read expert ID from ids tensor
        expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    }

    // Compute src1 and dst offsets
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1  = id;
    const int64_t i2  = iid1;

    const int blocks_per_row = ncols / QK_MXFP4;

    const bool      use_expert_ptrs = (expert_ptrs != nullptr);
    const uint8_t * base            = nullptr;
    if (use_expert_ptrs) {
        if (ids) {
            base = expert_ptrs[expert_id];
        } else {
            base = expert_ptrs[batch_idx];
        }
    } else {
        base = vx;
    }
    const int64_t qs_offset = use_expert_ptrs ? total_qs_size_per_expert : total_qs_size;

    // SoA layout for MXFP4: compute absolute row offset
    const int64_t abs_row          = use_expert_ptrs ? row : (expert_id * nrows_per_expert + row);
    const int64_t row_qs_offset    = abs_row * blocks_per_row * (QK_MXFP4 / 2);  // 16 bytes per block
    const int64_t row_scale_offset = qs_offset + abs_row * blocks_per_row;

    const uint8_t * qs_row    = base + row_qs_offset;
    const uint8_t * scale_row = base + row_scale_offset;

    // Q8_1 input: base pointer for this token (SoA layout)
    const char * y_base = (const char *) vy + i11 * nb11 + i12 * nb12;

    const int     lane_id         = item_ct1.get_local_id(2);
    constexpr int blocks_per_warp = WARP_SIZE;  // Each thread handles one block

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int b = lane_id;
    for (; b + 3 * blocks_per_warp < blocks_per_row; b += 4 * blocks_per_warp) {
        acc0 += mxfp4_soa_q8_1_block_dot<USE_WEIGHT_SCALE>(qs_row, scale_row, y_base, ncols_y, b);
        acc1 += mxfp4_soa_q8_1_block_dot<USE_WEIGHT_SCALE>(qs_row, scale_row, y_base, ncols_y, b + blocks_per_warp);
        acc2 += mxfp4_soa_q8_1_block_dot<USE_WEIGHT_SCALE>(qs_row, scale_row, y_base, ncols_y, b + 2 * blocks_per_warp);
        acc3 += mxfp4_soa_q8_1_block_dot<USE_WEIGHT_SCALE>(qs_row, scale_row, y_base, ncols_y, b + 3 * blocks_per_warp);
    }
    for (; b < blocks_per_row; b += blocks_per_warp) {
        acc0 += mxfp4_soa_q8_1_block_dot<USE_WEIGHT_SCALE>(qs_row, scale_row, y_base, ncols_y, b);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    acc       = sycl::reduce_over_group(item_ct1.get_sub_group(), acc, sycl::plus<float>());

    if (lane_id == 0) {
        float * dst_out = (float *) ((char *) dst + i1 * nb1 + i2 * nb2);
        dst_out[row]    = acc;
    }
}

static void mul_mat_vec_mxfp4_q8_1_soa_id_pair_kernel(const uint8_t * const * __restrict__ expert_ptrs_a,
                                                      const uint8_t * const * __restrict__ expert_ptrs_b,
                                                      const void * __restrict__ vy,
                                                      float * __restrict__ dst_a,
                                                      float * __restrict__ dst_b,
                                                      const int32_t * __restrict__ ids,
                                                      const int                ncols,
                                                      const int                ncols_y,
                                                      const int                nrows_per_expert,
                                                      const int                n_ids,
                                                      const int                n_tokens,
                                                      const int                ne11,
                                                      const int64_t            total_qs_size_per_expert,
                                                      const int64_t            ids_nb0,
                                                      const int64_t            ids_nb1,
                                                      const int64_t            nb11,
                                                      const int64_t            nb12,
                                                      const int64_t            nb1_a,
                                                      const int64_t            nb2_a,
                                                      const int64_t            nb1_b,
                                                      const int64_t            nb2_b,
                                                      const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    const int32_t expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    const int64_t i11       = id % ne11;
    const int64_t i12       = iid1;
    const int64_t i1        = id;
    const int64_t i2        = iid1;

    const int       blocks_per_row   = ncols / QK_MXFP4;
    const int64_t   row_qs_offset    = row * blocks_per_row * (QK_MXFP4 / 2);
    const int64_t   row_scale_offset = total_qs_size_per_expert + row * blocks_per_row;
    const uint8_t * base_a           = expert_ptrs_a[expert_id];
    const uint8_t * base_b           = expert_ptrs_b[expert_id];
    const uint8_t * qs_row_a         = base_a + row_qs_offset;
    const uint8_t * scale_row_a      = base_a + row_scale_offset;
    const uint8_t * qs_row_b         = base_b + row_qs_offset;
    const uint8_t * scale_row_b      = base_b + row_scale_offset;
    const char *    y_base           = (const char *) vy + i11 * nb11 + i12 * nb12;
    const int       lane_id          = item_ct1.get_local_id(2);
    constexpr int   blocks_per_warp  = WARP_SIZE;

    float acc_a0 = 0.0f;
    float acc_a1 = 0.0f;
    float acc_b0 = 0.0f;
    float acc_b1 = 0.0f;

    int b = lane_id;
    for (; b + blocks_per_warp < blocks_per_row; b += 2 * blocks_per_warp) {
        mxfp4_soa_q8_1_block_dot_pair(qs_row_a, scale_row_a, qs_row_b, scale_row_b, y_base, ncols_y, b, acc_a0, acc_b0);
        mxfp4_soa_q8_1_block_dot_pair(qs_row_a, scale_row_a, qs_row_b, scale_row_b, y_base, ncols_y,
                                      b + blocks_per_warp, acc_a1, acc_b1);
    }
    for (; b < blocks_per_row; b += blocks_per_warp) {
        mxfp4_soa_q8_1_block_dot_pair(qs_row_a, scale_row_a, qs_row_b, scale_row_b, y_base, ncols_y, b, acc_a0, acc_b0);
    }

    float acc_a = acc_a0 + acc_a1;
    float acc_b = acc_b0 + acc_b1;

    acc_a = sycl::reduce_over_group(item_ct1.get_sub_group(), acc_a, sycl::plus<float>());
    acc_b = sycl::reduce_over_group(item_ct1.get_sub_group(), acc_b, sycl::plus<float>());

    if (lane_id == 0) {
        float * dst_out_a = (float *) ((char *) dst_a + i1 * nb1_a + i2 * nb2_a);
        float * dst_out_b = (float *) ((char *) dst_b + i1 * nb1_b + i2 * nb2_b);
        dst_out_a[row]    = acc_a;
        dst_out_b[row]    = acc_b;
    }
}

static inline float mmvq_moe_apply_pair_glu(float       gate,
                                            float       up,
                                            const int   glu_op,
                                            const float alpha,
                                            const float limit) {
    if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
        const float gate_limited = sycl::fmin(gate, limit);
        const float up_limited   = sycl::fmax(sycl::fmin(up, limit), -limit);
        return (gate_limited / (1.0f + sycl::native::exp(-gate_limited * alpha))) * (1.0f + up_limited);
    }
    return (gate / (1.0f + sycl::native::exp(-gate))) * up;
}

template <int GLU_OP>
static __dpct_inline__ float mmvq_moe_apply_pair_glu(float gate, float up, const float alpha, const float limit) {
    if constexpr (GLU_OP == GGML_GLU_OP_SWIGLU_OAI) {
        const float gate_limited = sycl::fmin(gate, limit);
        const float up_limited   = sycl::fmax(sycl::fmin(up, limit), -limit);
        return (gate_limited / (1.0f + sycl::native::exp(-gate_limited * alpha))) * (1.0f + up_limited);
    } else {
        return (gate / (1.0f + sycl::native::exp(-gate))) * up;
    }
}

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
static bool mmvq_moe_direct_xmx_glu_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_SYCL_MOE_DIRECT_XMX_GLU");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

template <int TILES_N = 4>
static void mul_mat_vec_mxfp4_q8_1_soa_id_pair_glu_xmx_kernel(const uint8_t * const * __restrict__ gate_ptrs,
                                                              const uint8_t * const * __restrict__ up_ptrs,
                                                              const void * __restrict__ vy,
                                                              float * __restrict__ dst_glu,
                                                              const int32_t * __restrict__ ids,
                                                              const float * __restrict__ gate_bias,
                                                              const float * __restrict__ up_bias,
                                                              const int                       ncols,
                                                              const int                       ncols_y,
                                                              const int                       nrows_per_expert,
                                                              const int                       n_output_tiles,
                                                              const int                       n_tokens,
                                                              const int                       ne11,
                                                              const int64_t                   total_qs_size_per_expert,
                                                              const int64_t                   ids_nb0,
                                                              const int64_t                   ids_nb1,
                                                              const int64_t                   nb11,
                                                              const int64_t                   nb12,
                                                              const int64_t                   dst_nb1,
                                                              const int64_t                   dst_nb2,
                                                              const int64_t                   gate_bias_nb1,
                                                              const int64_t                   up_bias_nb1,
                                                              const int                       glu_op,
                                                              const float                     alpha,
                                                              const float                     limit,
                                                              const sycl::nd_item<1> &        item,
                                                              sycl::local_accessor<int8_t, 1> slm_token,
                                                              sycl::local_accessor<int8_t, 1> slm_gate_weights,
                                                              sycl::local_accessor<int8_t, 1> slm_up_weights,
                                                              sycl::local_accessor<int8_t, 1> slm_kvalues,
                                                              sycl::local_accessor<float, 1>  slm_gate_scales,
                                                              sycl::local_accessor<float, 1>  slm_up_scales,
                                                              sycl::local_accessor<float, 1>  slm_token_scale) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    constexpr int XMX_M       = 8;
    constexpr int XMX_N       = 16;
    constexpr int XMX_K       = 32;
    constexpr int SG_SIZE     = 16;
    constexpr int TILE_N      = TILES_N * XMX_N;
    constexpr int PACKED_SIZE = QK_MXFP4 / 2;

    auto sg   = item.get_sub_group();
    int  lane = sg.get_local_linear_id();

    const int work_group = item.get_group(0);
    const int batch_idx  = work_group / n_output_tiles;
    const int tile_idx   = work_group - batch_idx * n_output_tiles;
    const int col_start  = tile_idx * TILE_N;

    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    const int32_t expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    const int64_t i11       = id % ne11;
    const int64_t i12       = iid1;
    const int64_t i1        = id;
    const int64_t i2        = iid1;

    const uint8_t * gate_base = gate_ptrs[expert_id];
    const uint8_t * up_base   = up_ptrs[expert_id];
    if (!gate_base || !up_base) {
        return;
    }

    const char *        y_base       = (const char *) vy + i11 * nb11 + i12 * nb12;
    const int8_t *      y_qs         = (const int8_t *) y_base;
    const sycl::half2 * y_ds         = (const sycl::half2 *) (y_base + ncols_y);
    const int           num_k_blocks = ncols / QK_MXFP4;

    if (lane < 16) {
        slm_kvalues[lane] = kvalues_mxfp4[lane];
    }
    sycl::group_barrier(item.get_group());

    float gate_acc_float[TILE_N] = { 0.0f };
    float up_acc_float[TILE_N]   = { 0.0f };

    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> gate_acc[TILES_N];
    joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> up_acc[TILES_N];

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        for (int tn = 0; tn < TILES_N; ++tn) {
            joint_matrix_fill(sg, gate_acc[tn], 0);
            joint_matrix_fill(sg, up_acc[tn], 0);
        }

        for (int i = lane; i < XMX_M * XMX_K; i += SG_SIZE) {
            const int row = i / XMX_K;
            const int col = i - row * XMX_K;
            slm_token[i]  = row == 0 ? y_qs[k_block * XMX_K + col] : 0;
        }
        if (lane == 0) {
            slm_token_scale[0] = static_cast<float>(y_ds[k_block][0]);
        }

        for (int i = lane; i < TILE_N * XMX_K; i += SG_SIZE) {
            const int out_col_local = i / XMX_K;
            const int k_elem        = i - out_col_local * XMX_K;
            const int out_col       = col_start + out_col_local;

            int8_t gate_v = 0;
            int8_t up_v   = 0;
            float  gate_s = 0.0f;
            float  up_s   = 0.0f;
            if (out_col < nrows_per_expert) {
                const int64_t block_idx    = static_cast<int64_t>(out_col) * num_k_blocks + k_block;
                const int64_t qs_offset    = block_idx * PACKED_SIZE;
                const int64_t scale_offset = total_qs_size_per_expert + block_idx;
                const int     byte_idx     = k_elem % PACKED_SIZE;

                const uint8_t gate_packed = gate_base[qs_offset + byte_idx];
                const uint8_t up_packed   = up_base[qs_offset + byte_idx];
                const uint8_t gate_nibble = k_elem < PACKED_SIZE ? (gate_packed & 0xF) : (gate_packed >> 4);
                const uint8_t up_nibble   = k_elem < PACKED_SIZE ? (up_packed & 0xF) : (up_packed >> 4);

                gate_v = slm_kvalues[gate_nibble];
                up_v   = slm_kvalues[up_nibble];
                gate_s = sycl_e8m0_to_fp32_half(gate_base[scale_offset]);
                up_s   = sycl_e8m0_to_fp32_half(up_base[scale_offset]);
            }
            slm_gate_weights[out_col_local * XMX_K + k_elem] = gate_v;
            slm_up_weights[out_col_local * XMX_K + k_elem]   = up_v;
            if (k_elem == 0) {
                slm_gate_scales[out_col_local] = gate_s;
                slm_up_scales[out_col_local]   = up_s;
            }
        }
        sycl::group_barrier(item.get_group());

        joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
        auto                                                                           token_ptr =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                &slm_token[0]);
        joint_matrix_load(sg, mat_a, token_ptr, XMX_K);

        for (int tn = 0; tn < TILES_N; ++tn) {
            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> gate_b;
            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> up_b;

            auto gate_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    &slm_gate_weights[tn * XMX_N * XMX_K]);
            auto up_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    &slm_up_weights[tn * XMX_N * XMX_K]);
            joint_matrix_load(sg, gate_b, gate_ptr, XMX_K);
            joint_matrix_load(sg, up_b, up_ptr, XMX_K);
            joint_matrix_mad(sg, gate_acc[tn], mat_a, gate_b, gate_acc[tn]);
            joint_matrix_mad(sg, up_acc[tn], mat_a, up_b, up_acc[tn]);

            int32_t gate_row0 = 0;
            int     gate_slot = 0;
            joint_matrix_apply(sg, gate_acc[tn], [&](int32_t & elem) {
                if (gate_slot == 0) {
                    gate_row0 = elem;
                }
                ++gate_slot;
            });
            int32_t up_row0 = 0;
            int     up_slot = 0;
            joint_matrix_apply(sg, up_acc[tn], [&](int32_t & elem) {
                if (up_slot == 0) {
                    up_row0 = elem;
                }
                ++up_slot;
            });

            const float gate_scale = slm_token_scale[0] * slm_gate_scales[tn * XMX_N + lane];
            const float up_scale   = slm_token_scale[0] * slm_up_scales[tn * XMX_N + lane];
            gate_acc_float[tn * XMX_N + lane] += static_cast<float>(gate_row0) * gate_scale;
            up_acc_float[tn * XMX_N + lane] += static_cast<float>(up_row0) * up_scale;
        }
    }

    float * dst_out = (float *) ((char *) dst_glu + i1 * dst_nb1 + i2 * dst_nb2);
    for (int i = lane; i < TILE_N; i += SG_SIZE) {
        const int out_col = col_start + i;
        if (out_col < nrows_per_expert) {
            float gate = gate_acc_float[i];
            float up   = up_acc_float[i];
            if (gate_bias) {
                gate +=
                    *(const float *) ((const char *) gate_bias + expert_id * gate_bias_nb1 + out_col * sizeof(float));
            }
            if (up_bias) {
                up += *(const float *) ((const char *) up_bias + expert_id * up_bias_nb1 + out_col * sizeof(float));
            }
            dst_out[out_col] = mmvq_moe_apply_pair_glu(gate, up, glu_op, alpha, limit);
        }
    }
}

static void reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl(const void * const * gate_ptrs,
                                                                const void * const * up_ptrs,
                                                                const void *         vy,
                                                                float *              dst_glu,
                                                                const int32_t *      ids,
                                                                const float *        gate_bias,
                                                                const float *        up_bias,
                                                                const int            ncols,
                                                                const int            ncols_y,
                                                                const int            nrows_per_expert,
                                                                const int            total_batches,
                                                                const int            n_ids,
                                                                const int            n_tokens,
                                                                const int            ne11,
                                                                const int64_t        ids_nb0,
                                                                const int64_t        ids_nb1,
                                                                const int64_t        nb11,
                                                                const int64_t        nb12,
                                                                const int64_t        dst_nb1,
                                                                const int64_t        dst_nb2,
                                                                const int64_t        gate_bias_nb1,
                                                                const int64_t        up_bias_nb1,
                                                                const int            glu_op,
                                                                const float          alpha,
                                                                const float          limit,
                                                                dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_UNUSED(n_ids);
    constexpr int TILES_N = 1;
    constexpr int XMX_M   = 8;
    constexpr int XMX_N   = 16;
    constexpr int XMX_K   = 32;
    constexpr int SG_SIZE = 16;
    constexpr int TILE_N  = TILES_N * XMX_N;

    const int n_output_tiles = (nrows_per_expert + TILE_N - 1) / TILE_N;
    const int total_work     = total_batches * n_output_tiles;
    if (total_work <= 0) {
        return;
    }
    const int64_t total_qs_size_per_expert = (ncols / 2) * nrows_per_expert;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1> slm_token(sycl::range<1>(XMX_M * XMX_K), cgh);
        sycl::local_accessor<int8_t, 1> slm_gate_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        sycl::local_accessor<int8_t, 1> slm_up_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        sycl::local_accessor<int8_t, 1> slm_kvalues(sycl::range<1>(16), cgh);
        sycl::local_accessor<float, 1>  slm_gate_scales(sycl::range<1>(TILE_N), cgh);
        sycl::local_accessor<float, 1>  slm_up_scales(sycl::range<1>(TILE_N), cgh);
        sycl::local_accessor<float, 1>  slm_token_scale(sycl::range<1>(1), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_work * SG_SIZE), sycl::range<1>(SG_SIZE)),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                             mul_mat_vec_mxfp4_q8_1_soa_id_pair_glu_xmx_kernel<TILES_N>(
                                 (const uint8_t * const *) gate_ptrs, (const uint8_t * const *) up_ptrs, vy, dst_glu,
                                 ids, gate_bias, up_bias, ncols, ncols_y, nrows_per_expert, n_output_tiles, n_tokens,
                                 ne11, total_qs_size_per_expert, ids_nb0, ids_nb1, nb11, nb12, dst_nb1, dst_nb2,
                                 gate_bias_nb1, up_bias_nb1, glu_op, alpha, limit, item, slm_token, slm_gate_weights,
                                 slm_up_weights, slm_kvalues, slm_gate_scales, slm_up_scales, slm_token_scale);
                         });
    });
}
#endif

template <int GLU_OP, bool CACHE_Y_LOCAL, bool USE_WEIGHT_SCALE = true>
static void mul_mat_vec_mxfp4_q8_1_soa_id_pair_glu_kernel(const uint8_t * const * __restrict__ gate_ptrs,
                                                          const uint8_t * const * __restrict__ up_ptrs,
                                                          const void * __restrict__ vy,
                                                          float * __restrict__ dst_glu,
                                                          const int32_t * __restrict__ ids,
                                                          const float * __restrict__ gate_bias,
                                                          const float * __restrict__ up_bias,
                                                          const int                ncols,
                                                          const int                ncols_y,
                                                          const int                nrows_per_expert,
                                                          const int                n_ids,
                                                          const int                n_tokens,
                                                          const int                ne11,
                                                          const int64_t            total_qs_size_per_expert,
                                                          const int64_t            ids_nb0,
                                                          const int64_t            ids_nb1,
                                                          const int64_t            nb11,
                                                          const int64_t            nb12,
                                                          const int64_t            dst_nb1,
                                                          const int64_t            dst_nb2,
                                                          const int64_t            gate_bias_nb1,
                                                          const int64_t            up_bias_nb1,
                                                          const float              alpha,
                                                          const float              limit,
                                                          const sycl::nd_item<3> & item_ct1,
                                                          int * __restrict__ slm_y_qs,
                                                          sycl::half2 * __restrict__ slm_y_ds) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    const int32_t expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    const int64_t i11       = id % ne11;
    const int64_t i12       = iid1;
    const int64_t i1        = id;
    const int64_t i2        = iid1;

    const char * global_y_base = (const char *) vy + i11 * nb11 + i12 * nb12;
    const int    lane_id       = item_ct1.get_local_id(2);
    const int    local_row     = item_ct1.get_local_id(1);

    const int blocks_per_row = ncols / QK_MXFP4;
    if constexpr (CACHE_Y_LOCAL) {
        if (local_row == 0) {
            for (int b = lane_id; b < blocks_per_row; b += WARP_SIZE) {
                const int * q8_qs = (const int *) (global_y_base + b * QK8_1);
#pragma unroll
                for (int j = 0; j < QI8_1; ++j) {
                    slm_y_qs[b * QI8_1 + j] = q8_qs[j];
                }
                slm_y_ds[b] = *((const sycl::half2 *) (global_y_base + ncols_y + b * sizeof(sycl::half2)));
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (row >= nrows_per_expert) {
        return;
    }

    const int64_t   row_qs_offset    = row * blocks_per_row * (QK_MXFP4 / 2);
    const int64_t   row_scale_offset = total_qs_size_per_expert + row * blocks_per_row;
    const uint8_t * gate_base        = gate_ptrs[expert_id];
    const uint8_t * up_base          = up_ptrs[expert_id];
    const uint8_t * gate_qs_row      = gate_base + row_qs_offset;
    const uint8_t * gate_scale_row   = gate_base + row_scale_offset;
    const uint8_t * up_qs_row        = up_base + row_qs_offset;
    const uint8_t * up_scale_row     = up_base + row_scale_offset;
    constexpr int   blocks_per_warp  = WARP_SIZE;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    int b = lane_id;
    for (; b + blocks_per_warp < blocks_per_row; b += 2 * blocks_per_warp) {
        if constexpr (CACHE_Y_LOCAL) {
            mxfp4_soa_q8_1_block_dot_pair_slm<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                                slm_y_qs, slm_y_ds, QI8_1, b, gate_acc, up_acc);
            mxfp4_soa_q8_1_block_dot_pair_slm<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                                slm_y_qs, slm_y_ds, QI8_1, b + blocks_per_warp,
                                                                gate_acc, up_acc);
        } else {
            mxfp4_soa_q8_1_block_dot_pair<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                            global_y_base, ncols_y, b, gate_acc, up_acc);
            mxfp4_soa_q8_1_block_dot_pair<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                            global_y_base, ncols_y, b + blocks_per_warp, gate_acc,
                                                            up_acc);
        }
    }
    for (; b < blocks_per_row; b += blocks_per_warp) {
        if constexpr (CACHE_Y_LOCAL) {
            mxfp4_soa_q8_1_block_dot_pair_slm<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                                slm_y_qs, slm_y_ds, QI8_1, b, gate_acc, up_acc);
        } else {
            mxfp4_soa_q8_1_block_dot_pair<USE_WEIGHT_SCALE>(gate_qs_row, gate_scale_row, up_qs_row, up_scale_row,
                                                            global_y_base, ncols_y, b, gate_acc, up_acc);
        }
    }

    gate_acc = sycl::reduce_over_group(item_ct1.get_sub_group(), gate_acc, sycl::plus<float>());
    up_acc   = sycl::reduce_over_group(item_ct1.get_sub_group(), up_acc, sycl::plus<float>());

    if (lane_id == 0) {
        if (gate_bias) {
            gate_acc += *(const float *) ((const char *) gate_bias + expert_id * gate_bias_nb1 + row * sizeof(float));
        }
        if (up_bias) {
            up_acc += *(const float *) ((const char *) up_bias + expert_id * up_bias_nb1 + row * sizeof(float));
        }
        float * dst_out = (float *) ((char *) dst_glu + i1 * dst_nb1 + i2 * dst_nb2);
        dst_out[row]    = mmvq_moe_apply_pair_glu<GLU_OP>(gate_acc, up_acc, alpha, limit);
    }
}

static void mul_mat_vec_mxfp4_q8_1_coalesced_id_kernel(
    const uint8_t * __restrict__ vx,
    const uint8_t * const * __restrict__ expert_ptrs,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int32_t * __restrict__ ids,
    const int                ncols,
    const int                ncols_y,  // Q8_1 row size (for SoA ds offset)
    const int                nrows_per_expert,
    const int                n_ids,
    const int                n_tokens,
    const int                ne11,
    const int64_t            total_qs_size,             // Offset to MXFP4 scale region
    const int64_t            total_qs_size_per_expert,  // Offset to MXFP4 scale region (per-expert layout)
    const int64_t            ids_nb0,
    const int64_t            ids_nb1,
    const int64_t            nb11,
    const int64_t            nb12,
    const int64_t            nb1,
    const int64_t            nb2,
    const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row       = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    const int id   = batch_idx / n_tokens;
    const int iid1 = batch_idx - id * n_tokens;

    int32_t expert_id = -1;
    if (ids) {
        expert_id = *(const int32_t *) ((const char *) ids + iid1 * ids_nb1 + id * ids_nb0);
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1  = id;
    const int64_t i2  = iid1;

    const int blocks_per_row = ncols / QK_MXFP4;

    const bool      use_expert_ptrs = (expert_ptrs != nullptr);
    const uint8_t * base            = nullptr;
    if (use_expert_ptrs) {
        base = ids ? expert_ptrs[expert_id] : expert_ptrs[batch_idx];
    } else {
        base = vx;
    }
    const int64_t qs_offset = use_expert_ptrs ? total_qs_size_per_expert : total_qs_size;

    const int64_t abs_row          = use_expert_ptrs ? row : (expert_id * nrows_per_expert + row);
    const int64_t row_qs_offset    = abs_row * (ncols / 2);
    const int64_t row_scale_offset = qs_offset + abs_row * blocks_per_row;

    const uint8_t * x_qs = base + row_qs_offset;
    const uint8_t * x_e  = base + row_scale_offset;

    const char *        y_base = (const char *) vy + i11 * nb11 + i12 * nb12;
    const int8_t *      y_qs   = (const int8_t *) y_base;
    const sycl::half2 * y_ds   = (const sycl::half2 *) (y_base + ncols_y);

    const int     lane_id       = item_ct1.get_local_id(2);
    constexpr int TILE_BLOCKS   = MMVQ_COALESCED_TILE_BLOCKS;
    const int     tiles_per_row = blocks_per_row / TILE_BLOCKS;
    const int     word_stride   = TILE_BLOCKS * 4;

    float acc = 0.0f;

    for (int tile = 0; tile < tiles_per_row; ++tile) {
        const int tile_base = row_qs_offset + tile * (TILE_BLOCKS * (QK_MXFP4 / 2));

        for (int block_in_tile = lane_id; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
            const int block_idx = tile * TILE_BLOCKS + block_in_tile;

            const uint8_t e8m0  = x_e[block_idx];
            const float   scale = ggml_sycl_e8m0_to_fp32(e8m0) * 0.5f;

            const int y_block       = block_idx;
            const int y_base_offset = y_block * QK8_1;

            for (int half = 0; half < 2; ++half) {
                const int word_base    = half * (2 * word_stride);
                const int word0_offset = word_base + block_in_tile * 4;
                const int word1_offset = word_base + word_stride + block_in_tile * 4;

                const int v0 = *((const int *) (x_qs + tile_base + word0_offset));
                const int v1 = *((const int *) (x_qs + tile_base + word1_offset));

                const sycl::int2 dq0 = get_int_from_table_16(v0, kvalues_mxfp4);
                const sycl::int2 dq1 = get_int_from_table_16(v1, kvalues_mxfp4);

                const int y_offset = half * 8;
                const int u0       = get_int_from_int8_aligned(y_qs + y_base_offset, y_offset / 4);
                const int u1       = get_int_from_int8_aligned(y_qs + y_base_offset, y_offset / 4 + 4);
                const int u2       = get_int_from_int8_aligned(y_qs + y_base_offset, y_offset / 4 + 1);
                const int u3       = get_int_from_int8_aligned(y_qs + y_base_offset, y_offset / 4 + 5);

                int sumi = 0;
                sumi     = ggml_sycl_dp4a(dq0.x(), u0, sumi);
                sumi     = ggml_sycl_dp4a(dq0.y(), u1, sumi);
                sumi     = ggml_sycl_dp4a(dq1.x(), u2, sumi);
                sumi     = ggml_sycl_dp4a(dq1.y(), u3, sumi);

                const float d8 = (float) y_ds[y_block][0];
                acc += scale * d8 * sumi;
            }
        }
    }

    acc = sycl::reduce_over_group(item_ct1.get_sub_group(), acc, sycl::plus<float>());

    if (lane_id == 0) {
        float * dst_out = (float *) ((char *) dst + i1 * nb1 + i2 * nb2);
        dst_out[row]    = acc;
    }
}

// MoE dispatch: MXFP4 SoA layout with expert routing
// Both MXFP4 weights and Q8_1 inputs must be in SoA layout
template <int MOE_MMV_Y, bool USE_WEIGHT_SCALE = true>
static void reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows(const void *         vx,
                                                        const void * const * expert_ptrs,
                                                        const void *         vy,
                                                        float *              dst,
                                                        const int32_t *      ids,
                                                        const int            ncols,
                                                        const int       ncols_y,  // Q8_1 row size (for SoA ds offset)
                                                        const int       nrows_per_expert,
                                                        const int       num_experts,
                                                        const int       total_batches,
                                                        const int       n_ids,
                                                        const int       n_tokens,
                                                        const int       ne11,
                                                        const int64_t   ids_nb0,
                                                        const int64_t   ids_nb1,
                                                        const int64_t   nb11,
                                                        const int64_t   nb12,
                                                        const int64_t   nb1,
                                                        const int64_t   nb2,
                                                        dpct::queue_ptr stream,
                                                        sycl::event *   event_out = nullptr) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    static_assert(MOE_MMV_Y > 0, "MOE_MMV_Y must be positive");
    static_assert(MOE_MMV_Y * WARP_SIZE <= 1024, "MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + MOE_MMV_Y - 1) / MOE_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, MOE_MMV_Y, WARP_SIZE);

    // SoA layout for MXFP4: qs_size = (ncols / 2) * total_rows
    const int64_t total_rows               = (int64_t) nrows_per_expert * num_experts;
    const int64_t total_qs_size            = (ncols / 2) * total_rows;
    const int64_t total_qs_size_per_expert = (ncols / 2) * nrows_per_expert;

    sycl::event ev = stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_mxfp4_q8_1_soa_id_kernel<USE_WEIGHT_SCALE>(
                                 (const uint8_t *) vx, (const uint8_t * const *) expert_ptrs, vy, dst, ids, ncols,
                                 ncols_y, nrows_per_expert, n_ids, n_tokens, ne11, total_qs_size,
                                 total_qs_size_per_expert, ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
    if (event_out) {
        *event_out = ev;
    }
}

static void reorder_mul_mat_vec_mxfp4_q8_1_id_sycl(const void *         vx,
                                                   const void * const * expert_ptrs,
                                                   const void *         vy,
                                                   float *              dst,
                                                   const int32_t *      ids,
                                                   const int            ncols,
                                                   const int            ncols_y,  // Q8_1 row size (for SoA ds offset)
                                                   const int            nrows_per_expert,
                                                   const int            num_experts,
                                                   const int            total_batches,
                                                   const int            n_ids,
                                                   const int            n_tokens,
                                                   const int            ne11,
                                                   const int64_t        ids_nb0,
                                                   const int64_t        ids_nb1,
                                                   const int64_t        nb11,
                                                   const int64_t        nb12,
                                                   const int64_t        nb1,
                                                   const int64_t        nb2,
                                                   dpct::queue_ptr      stream,
                                                   sycl::event *        event_out = nullptr) {
    reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<GGML_SYCL_MOE_MMV_Y>(
        vx, expert_ptrs, vy, dst, ids, ncols, ncols_y, nrows_per_expert, num_experts, total_batches, n_ids, n_tokens,
        ne11, ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, stream, event_out);
}

static void reorder_mul_mat_vec_mxfp4_q8_1_id_pair_sycl(const void * const * expert_ptrs_a,
                                                        const void * const * expert_ptrs_b,
                                                        const void *         vy,
                                                        float *              dst_a,
                                                        float *              dst_b,
                                                        const int32_t *      ids,
                                                        const int            ncols,
                                                        const int            ncols_y,
                                                        const int            nrows_per_expert,
                                                        const int            total_batches,
                                                        const int            n_ids,
                                                        const int            n_tokens,
                                                        const int            ne11,
                                                        const int64_t        ids_nb0,
                                                        const int64_t        ids_nb1,
                                                        const int64_t        nb11,
                                                        const int64_t        nb12,
                                                        const int64_t        nb1_a,
                                                        const int64_t        nb2_a,
                                                        const int64_t        nb1_b,
                                                        const int64_t        nb2_b,
                                                        dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr int moe_mmv_y = GGML_SYCL_MOE_MMV_Y;
    static_assert(moe_mmv_y * WARP_SIZE <= 1024, "GGML_SYCL_MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + moe_mmv_y - 1) / moe_mmv_y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, moe_mmv_y, WARP_SIZE);

    const int64_t total_qs_size_per_expert = (ncols / 2) * nrows_per_expert;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_mxfp4_q8_1_soa_id_pair_kernel(
                                 (const uint8_t * const *) expert_ptrs_a, (const uint8_t * const *) expert_ptrs_b, vy,
                                 dst_a, dst_b, ids, ncols, ncols_y, nrows_per_expert, n_ids, n_tokens, ne11,
                                 total_qs_size_per_expert, ids_nb0, ids_nb1, nb11, nb12, nb1_a, nb2_a, nb1_b, nb2_b,
                                 item_ct1);
                         });
    });
}

template <int MOE_MMV_Y, bool USE_WEIGHT_SCALE = true>
static void reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows(const void * const * gate_ptrs,
                                                                 const void * const * up_ptrs,
                                                                 const void *         vy,
                                                                 float *              dst_glu,
                                                                 const int32_t *      ids,
                                                                 const float *        gate_bias,
                                                                 const float *        up_bias,
                                                                 const int            ncols,
                                                                 const int            ncols_y,
                                                                 const int            nrows_per_expert,
                                                                 const int            total_batches,
                                                                 const int            n_ids,
                                                                 const int            n_tokens,
                                                                 const int            ne11,
                                                                 const int64_t        ids_nb0,
                                                                 const int64_t        ids_nb1,
                                                                 const int64_t        nb11,
                                                                 const int64_t        nb12,
                                                                 const int64_t        dst_nb1,
                                                                 const int64_t        dst_nb2,
                                                                 const int64_t        gate_bias_nb1,
                                                                 const int64_t        up_bias_nb1,
                                                                 const int            glu_op,
                                                                 const float          alpha,
                                                                 const float          limit,
                                                                 bool                 cache_y_local,
                                                                 dpct::queue_ptr      stream,
                                                                 sycl::event *        event_out = nullptr) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    static_assert(MOE_MMV_Y > 0, "MOE_MMV_Y must be positive");
    static_assert(MOE_MMV_Y * WARP_SIZE <= 1024, "MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + MOE_MMV_Y - 1) / MOE_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, MOE_MMV_Y, WARP_SIZE);

    const int64_t total_qs_size_per_expert = (ncols / 2) * nrows_per_expert;
    const int     blocks_per_row           = ncols / QK_MXFP4;
    sycl::event   kernel_event;
    auto          submit_glu = [&](auto glu_tag, auto cache_tag) -> sycl::event {
        constexpr int  GLU_OP        = decltype(glu_tag)::value;
        constexpr bool CACHE_Y_LOCAL = decltype(cache_tag)::value;
        return stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<int, 1> slm_y_qs(
                sycl::range<1>(CACHE_Y_LOCAL ? static_cast<size_t>(blocks_per_row * QI8_1) : 1), cgh);
            sycl::local_accessor<sycl::half2, 1> slm_y_ds(
                sycl::range<1>(CACHE_Y_LOCAL ? static_cast<size_t>(blocks_per_row) : 1), cgh);
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                                      [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_mxfp4_q8_1_soa_id_pair_glu_kernel<GLU_OP, CACHE_Y_LOCAL, USE_WEIGHT_SCALE>(
                                     (const uint8_t * const *) gate_ptrs, (const uint8_t * const *) up_ptrs, vy,
                                     dst_glu, ids, gate_bias, up_bias, ncols, ncols_y, nrows_per_expert, n_ids,
                                     n_tokens, ne11, total_qs_size_per_expert, ids_nb0, ids_nb1, nb11, nb12, dst_nb1,
                                     dst_nb2, gate_bias_nb1, up_bias_nb1, alpha, limit, item_ct1,
                                     SYCL_LOCAL_ACC_PTR(slm_y_qs), SYCL_LOCAL_ACC_PTR(slm_y_ds));
                             });
        });
    };
    if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
        if (cache_y_local) {
            kernel_event = submit_glu(std::integral_constant<int, GGML_GLU_OP_SWIGLU_OAI>{}, std::true_type{});
        } else {
            kernel_event = submit_glu(std::integral_constant<int, GGML_GLU_OP_SWIGLU_OAI>{}, std::false_type{});
        }
    } else {
        if (cache_y_local) {
            kernel_event = submit_glu(std::integral_constant<int, GGML_GLU_OP_SWIGLU>{}, std::true_type{});
        } else {
            kernel_event = submit_glu(std::integral_constant<int, GGML_GLU_OP_SWIGLU>{}, std::false_type{});
        }
    }
    if (event_out) {
        *event_out = kernel_event;
    }
}

static void reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl(const void * const * gate_ptrs,
                                                            const void * const * up_ptrs,
                                                            const void *         vy,
                                                            float *              dst_glu,
                                                            const int32_t *      ids,
                                                            const float *        gate_bias,
                                                            const float *        up_bias,
                                                            const int            ncols,
                                                            const int            ncols_y,
                                                            const int            nrows_per_expert,
                                                            const int            total_batches,
                                                            const int            n_ids,
                                                            const int            n_tokens,
                                                            const int            ne11,
                                                            const int64_t        ids_nb0,
                                                            const int64_t        ids_nb1,
                                                            const int64_t        nb11,
                                                            const int64_t        nb12,
                                                            const int64_t        dst_nb1,
                                                            const int64_t        dst_nb2,
                                                            const int64_t        gate_bias_nb1,
                                                            const int64_t        up_bias_nb1,
                                                            const int            glu_op,
                                                            const float          alpha,
                                                            const float          limit,
                                                            dpct::queue_ptr      stream,
                                                            sycl::event *        event_out = nullptr) {
    GGML_UNUSED(nb11);
    const bool cache_y_local = false;
    reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<GGML_SYCL_MOE_PAIR_GLU_MMV_Y>(
        gate_ptrs, up_ptrs, vy, dst_glu, ids, gate_bias, up_bias, ncols, ncols_y, nrows_per_expert, total_batches,
        n_ids, n_tokens, ne11, ids_nb0, ids_nb1, nb11, nb12, dst_nb1, dst_nb2, gate_bias_nb1, up_bias_nb1, glu_op,
        alpha, limit, cache_y_local, stream, event_out);
}

// MoE dispatch: MXFP4 Coalesced layout with expert routing
// MXFP4 weights are coalesced; Q8_1 inputs remain SoA (qs then ds per row)
static void coalesced_mul_mat_vec_mxfp4_q8_1_id_sycl(const void *         vx,
                                                     const void * const * expert_ptrs,
                                                     const void *         vy,
                                                     float *              dst,
                                                     const int32_t *      ids,
                                                     const int            ncols,
                                                     const int            ncols_y,  // Q8_1 row size (for SoA ds offset)
                                                     const int            nrows_per_expert,
                                                     const int            num_experts,
                                                     const int            total_batches,
                                                     const int            n_ids,
                                                     const int            n_tokens,
                                                     const int            ne11,
                                                     const int64_t        ids_nb0,
                                                     const int64_t        ids_nb1,
                                                     const int64_t        nb11,
                                                     const int64_t        nb12,
                                                     const int64_t        nb1,
                                                     const int64_t        nb2,
                                                     dpct::queue_ptr      stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    GGML_ASSERT((ncols / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);
    constexpr int moe_mmv_y = GGML_SYCL_MOE_MMV_Y;
    static_assert(moe_mmv_y * WARP_SIZE <= 1024, "GGML_SYCL_MOE_MMV_Y exceeds SYCL work-group limit");
    const int            block_num_z = (nrows_per_expert + moe_mmv_y - 1) / moe_mmv_y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, moe_mmv_y, WARP_SIZE);

    const int64_t total_rows               = (int64_t) nrows_per_expert * num_experts;
    const int64_t total_qs_size            = (ncols / 2) * total_rows;
    const int64_t total_qs_size_per_expert = (ncols / 2) * nrows_per_expert;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_mxfp4_q8_1_coalesced_id_kernel(
                                 (const uint8_t *) vx, (const uint8_t * const *) expert_ptrs, vy, dst, ids, ncols,
                                 ncols_y, nrows_per_expert, n_ids, n_tokens, ne11, total_qs_size,
                                 total_qs_size_per_expert, ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

static bool ggml_sycl_moe_ensure_compact_storage(ggml_backend_sycl_context & ctx,
                                                 ggml_tensor_extra_gpu *     extra,
                                                 int64_t                     total_batches,
                                                 bool                        allow_alloc) {
    if (!extra || total_batches <= 0) {
        return false;
    }
    if (ctx.device < 0 || ctx.device >= GGML_SYCL_MAX_DEVICES) {
        return false;
    }

    sycl::queue * stream = ctx.stream();
    const size_t  bytes  = static_cast<size_t>(total_batches) * sizeof(void *);

    if (extra->moe_expert_ptrs_compact_device[ctx.device] != nullptr &&
        extra->moe_expert_ptrs_compact_capacity[ctx.device] >= bytes) {
        extra->moe_expert_ptrs_compact_size[ctx.device] = bytes;
    } else {
        if (extra->moe_expert_ptrs_compact_device[ctx.device] != nullptr) {
            if (!extra->moe_expert_ptrs_compact_from_prealloc[ctx.device]) {
                sycl::free(extra->moe_expert_ptrs_compact_device[ctx.device], *stream);
            }
            extra->moe_expert_ptrs_compact_device[ctx.device]        = nullptr;
            extra->moe_expert_ptrs_compact_capacity[ctx.device]      = 0;
            extra->moe_expert_ptrs_compact_size[ctx.device]          = 0;
            extra->moe_expert_ptrs_compact_from_prealloc[ctx.device] = false;
        }

        void * compact = ggml_sycl::moe_get_compact_ptrs(ctx.device, bytes);
        if (compact) {
            extra->moe_expert_ptrs_compact_device[ctx.device]        = compact;
            extra->moe_expert_ptrs_compact_capacity[ctx.device]      = bytes;
            extra->moe_expert_ptrs_compact_size[ctx.device]          = bytes;
            extra->moe_expert_ptrs_compact_from_prealloc[ctx.device] = true;
        } else if (!allow_alloc) {
            return false;
        } else {
            compact = ggml_sycl_malloc_device(bytes, *stream, "mmvq_compact");
            if (!compact) {
                GGML_LOG_ERROR("[MOE] Failed to allocate compact pointer list (%zu bytes)\n", bytes);
                return false;
            }
            extra->moe_expert_ptrs_compact_device[ctx.device]        = compact;
            extra->moe_expert_ptrs_compact_capacity[ctx.device]      = bytes;
            extra->moe_expert_ptrs_compact_size[ctx.device]          = bytes;
            extra->moe_expert_ptrs_compact_from_prealloc[ctx.device] = false;
        }
    }

    if (extra->moe_expert_ptrs_missing_device[ctx.device] == nullptr) {
        int * missing = ggml_sycl::moe_get_compact_missing_flag(ctx.device);
        if (missing) {
            extra->moe_expert_ptrs_missing_device[ctx.device]        = missing;
            extra->moe_expert_ptrs_missing_from_prealloc[ctx.device] = true;
        } else if (!allow_alloc) {
            return false;
        } else {
            missing = ggml_sycl_malloc_device_t<int>(1, *stream, "mmvq_missing");
            if (!missing) {
                GGML_LOG_ERROR("[MOE] Failed to allocate compact list missing flag\n");
                return false;
            }
            extra->moe_expert_ptrs_missing_device[ctx.device]        = missing;
            extra->moe_expert_ptrs_missing_from_prealloc[ctx.device] = false;
        }
    }

    return true;
}

bool ggml_sycl_moe_prepare_compact_list(ggml_backend_sycl_context & ctx,
                                        const ggml_tensor *         src0,
                                        int64_t                     total_batches,
                                        bool                        allow_alloc) {
    if (!src0) {
        return false;
    }

    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra) {
        return false;
    }

    return ggml_sycl_moe_ensure_compact_storage(ctx, extra, total_batches, allow_alloc);
}

static sycl::event ggml_sycl_build_moe_compact_list(sycl::queue &                    queue,
                                                    void **                          compact_ptrs,
                                                    const void * const *             expert_ptrs,
                                                    const int32_t *                  ids,
                                                    int64_t                          n_ids,
                                                    int64_t                          n_tokens,
                                                    int64_t                          n_experts,
                                                    int64_t                          ids_nb0,
                                                    int64_t                          ids_nb1,
                                                    int *                            missing_flag,
                                                    const std::vector<sycl::event> & deps) {
    if (!compact_ptrs || !expert_ptrs || !ids || n_ids <= 0 || n_tokens <= 0) {
        return sycl::event();
    }

    const size_t total_batches = static_cast<size_t>(n_ids * n_tokens);
    sycl::event  clear_event;
    if (missing_flag) {
        clear_event = queue.submit([&](sycl::handler & cgh) {
            if (!deps.empty()) {
                cgh.depends_on(deps);
            }
            cgh.single_task([=]() { *missing_flag = 0; });
        });
    }

    std::vector<sycl::event> all_deps = deps;
    if (missing_flag) {
        all_deps.push_back(clear_event);
    }

    return queue.submit([&](sycl::handler & cgh) {
        if (!all_deps.empty()) {
            cgh.depends_on(all_deps);
        }
        cgh.parallel_for(sycl::range<1>(total_batches), [=](sycl::id<1> idx) {
            const int64_t batch_idx = static_cast<int64_t>(idx[0]);
            const int64_t id        = batch_idx / n_tokens;
            const int64_t iid1      = batch_idx - id * n_tokens;
            const char *  ids_base  = reinterpret_cast<const char *>(ids);
            const int32_t expert_id = *(const int32_t *) (ids_base + iid1 * ids_nb1 + id * ids_nb0);

            void * expert_ptr = nullptr;
            if (expert_id >= 0 && expert_id < n_experts) {
                expert_ptr = const_cast<void *>(expert_ptrs[expert_id]);
            }

            compact_ptrs[batch_idx] = expert_ptr;

            if (missing_flag && expert_ptr == nullptr) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    missing_ref(*missing_flag);
                missing_ref.store(1);
            }
        });
    });
}

#ifdef GGML_SYCL_GRAPH
// Pre-allocate Q8_1 buffers for all MUL_MAT_ID operations before graph recording.
// This must be called during decode phase, before graph recording starts.
// MUL_MAT_ID normally allocates Q8_1 buffers dynamically via ggml_sycl_pool_alloc,
// which is incompatible with SYCL graph recording.

void ggml_sycl_moe_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    queue_ptr stream = ctx.stream();

    // Count MUL_MAT_ID nodes and find max dimensions
    int              moe_count      = 0;
    int64_t          max_ne10       = 0;
    int64_t          max_src1_rows  = 0;
    ggml_type        demand_type    = GGML_TYPE_COUNT;
    ggml_layout_mode demand_layout  = GGML_LAYOUT_AOS;
    int              demand_layer   = -1;
    int              demand_experts = 0;
    int              demand_used    = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        const ggml_tensor * src0 = node->src[0];  // Expert weights
        const ggml_tensor * src1 = node->src[1];  // Input activations

        // Only count graph-compatible types
        if (!ggml_is_quantized(src0->type)) {
            continue;
        }
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
                break;
            default:
                continue;  // Skip unsupported types
        }

        moe_count++;

        // Track max dimensions
        const int64_t ne10         = src1->ne[0];
        const int64_t ne11         = src1->ne[1];
        const int64_t ne12         = src1->ne[2];
        const int64_t total_rows   = ne11 * ne12;
        const bool    host_weights = ggml_sycl_is_host_resident_weight(src0, ctx.stream());
        const auto    layout       = ggml_sycl_select_moe_mmvq_layout(src0, ctx.device, host_weights);

        if (ne10 > max_ne10) {
            max_ne10 = ne10;
        }
        if (total_rows > max_src1_rows) {
            max_src1_rows = total_rows;
        }
        demand_type    = src0->type;
        demand_layout  = layout;
        demand_layer   = std::max(demand_layer, ggml_sycl::expert_layer_from_tensor_name(src0->name));
        demand_experts = std::max<int>(demand_experts, static_cast<int>(src0->ne[2]));
        demand_used    = std::max<int>(demand_used, static_cast<int>(node->src[2] ? node->src[2]->ne[0] : 0));
    }

    if (moe_count == 0) {
        return;  // No MoE operations
    }

    ggml_sycl::moe_q8_1_scratch_shape shape;
    shape.weight_type    = demand_type;
    shape.layout         = demand_layout;
    shape.input_cols     = max_ne10;
    shape.input_rows     = max_src1_rows;
    shape.graph_op_count = moe_count;
    shape.device_id      = ctx.device;
    shape.segment_id     = -1;
    shape.layer_id       = demand_layer;
    shape.n_experts      = demand_experts;
    shape.n_expert_used  = demand_used;

    const auto   demand      = ggml_sycl::unified_cache_plan_moe_q8_1_scratch(shape);
    const size_t buffer_size = demand.bytes_per_buffer;

    GGML_SYCL_DEBUG("[MOE-GRAPH] Pre-allocating %d Q8_1 buffers, %zu bytes each (ne10=%lld, rows=%lld)\n", moe_count,
                    buffer_size, (long long) max_ne10, (long long) max_src1_rows);

    // Skip if already initialized with sufficient stable scratch buffers.
    if (ctx.moe_buffers.initialized) {
        const bool count_ok = ctx.moe_buffers.q8_1_buffers.size() >= static_cast<size_t>(moe_count);
        const bool size_ok  = ctx.moe_buffers.max_ne10 >= max_ne10 && ctx.moe_buffers.max_src1_rows >= max_src1_rows;
        const bool owner_ok = ctx.moe_buffers.q8_1_owner.ptr && ctx.moe_buffers.q8_1_bytes >= demand.total_bytes &&
                              ctx.moe_buffers.q8_1_base == ctx.moe_buffers.q8_1_owner.ptr;
        if (count_ok && size_ok && owner_ok) {
            ctx.moe_buffers.reset_usage();
            return;
        }
        ctx.moe_buffers.free_buffers(stream);
    }

    ggml_sycl::alloc_handle graph_owner{};
    if (!ggml_sycl::unified_cache_allocate_moe_q8_1_graph_scratch(ctx.device, demand, &graph_owner)) {
        GGML_LOG_ERROR(
            "[MOE-GRAPH] Failed to allocate owned Q8_1 graph scratch "
            "(requested=%zu bytes, buffers=%d, bytes_per_buffer=%zu, device=%d, layout=%d, dtype=%d, "
            "layer=%d, experts=%d, used=%d)\n",
            demand.total_bytes, demand.buffer_count, demand.bytes_per_buffer, ctx.device, demand.layout,
            demand.weight_type, demand.layer_id, demand.n_experts, demand.n_expert_used);
        return;
    }

    // Allocate buffers
    ctx.moe_buffers.q8_1_buffers.resize(moe_count);
    ctx.moe_buffers.q8_1_sizes.resize(moe_count);
    ctx.moe_buffers.q8_1_owner = graph_owner;
    ctx.moe_buffers.q8_1_base  = graph_owner.ptr;
    ctx.moe_buffers.q8_1_bytes = graph_owner.size;
    graph_owner                = {};

    uint8_t * graph_base = static_cast<uint8_t *>(ctx.moe_buffers.q8_1_base);
    for (int i = 0; i < moe_count; i++) {
        ctx.moe_buffers.q8_1_buffers[i] = graph_base + static_cast<size_t>(i) * demand.aligned_bytes_per_buffer;
        ctx.moe_buffers.q8_1_sizes[i]   = buffer_size;

        if (!ctx.moe_buffers.q8_1_buffers[i]) {
            GGML_LOG_ERROR(
                "[MOE-GRAPH] Failed to slice Q8_1 buffer %d from owned graph scratch "
                "(requested=%zu bytes, total=%zu bytes, device=%d, layout=%d, "
                "dtype=%d, layer=%d, experts=%d, used=%d)\n",
                i, buffer_size, demand.total_bytes, ctx.device, demand.layout, demand.weight_type, demand.layer_id,
                demand.n_experts, demand.n_expert_used);
            ctx.moe_buffers.free_buffers(stream);
            return;
        }
    }

    ctx.moe_buffers.max_ne10      = max_ne10;
    ctx.moe_buffers.max_src1_rows = max_src1_rows;
    ctx.moe_buffers.initialized   = true;
    ctx.moe_buffers.reset_usage();

    // Wait for allocations to complete
    stream->wait();

    GGML_SYCL_DEBUG("[MOE-GRAPH] Pre-allocated %d buffers successfully\n", moe_count);
}
#endif

// ---------------------------------------------------------------------------
// Batched MMVQ dispatch for MoE hybrid GPU path.
//
// Replaces per-expert ggml_sycl_mul_mat() loop with a single kernel
// submission.  Q8_1 quantization of src1 runs once and is shared across
// all GPU-staged experts.
//
// For entries dispatched to CPU (cache misses), we fill the pointer
// table with a safe dummy pointer so the kernel writes *something* to
// those dst slots.  The CPU path overwrites those slots afterwards.
// ---------------------------------------------------------------------------
bool mmvq_moe_batched_dispatch(ggml_backend_sycl_context &   ctx,
                               const ggml_tensor *           src0,
                               const ggml_tensor *           src1,
                               ggml_tensor *                 dst,
                               const void * const *          expert_ptrs_device,
                               const int32_t *               gpu_expert_ids,
                               const int64_t *               gpu_iid1s,
                               const int64_t *               gpu_ids,
                               int                           n_gpu_entries,
                               int                           n_experts,
                               int64_t                       n_ids,
                               layout_mode                   layout,
                               const int32_t *               direct_ids_device,
                               int64_t                       direct_ids_nb0,
                               int64_t                       direct_ids_nb1,
                               const ggml_sycl::mem_handle * src1_handle_override) {
    if (n_gpu_entries <= 0 || !expert_ptrs_device) {
        return false;
    }

    // Only handle quantized types with _id kernels
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    const queue_ptr stream     = ctx.stream();
    const bool      tg_profile = mmvq_moe_tg_profile_enabled() && src0->type == GGML_TYPE_MXFP4 && ne12 == 1;

    // --- Q8_1 quantization (shared across all experts) ---
    const int64_t ne10_padded   = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;

    // For decode (ne12 == 1), there is exactly 1 token row
    const int64_t total_src1_rows = ne11 * ne12;
    const size_t  required_size   = total_src1_rows * q8_1_row_size;

    const float * src1_d = static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(src1, ctx.device));
    float *       dst_d  = static_cast<float *>(ggml_sycl_resolve_tensor_ptr(dst, ctx.device));
    if (!src1_d || !dst_d) {
        GGML_SYCL_DEBUG("[MMVQ] Missing resolved ptrs for batched MoE dispatch (%s)\n", src0->name ? src0->name : "?");
        return false;
    }

    // Use SoA quantizer if weights are in reordered layout (both SoA and COALESCED kernels expect SoA Y)
    const bool     y_soa       = (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED);
    mxfp4_moe_role reuse_role  = mxfp4_moe_role::OTHER;
    int            reuse_layer = -1;
    const bool     reuse_candidate =
        mxfp4_moe_tg_reuse_candidate(src0, src0->type, layout, ne12, total_src1_rows, &reuse_role, &reuse_layer);
    const int             runtime_device = ctx.device;
    ggml_sycl::mem_handle src1_handle;
    if (src1_handle_override && src1_handle_override->valid()) {
        src1_handle        = *src1_handle_override;
        auto src1_resolved = src1_handle.resolve(runtime_device);
        if (!src1_resolved || src1_resolved.ptr != src1_d) {
            src1_handle = {};
        }
    }
    if (!src1_handle.valid()) {
        src1_handle =
            mxfp4_moe_tensor_mem_handle(src1, runtime_device, const_cast<float *>(src1_d), GGML_LAYOUT_AOS, true);
    }
    const bool reuse_q8 = reuse_candidate && mxfp4_moe_tg_reuse_can_use(runtime_device, reuse_layer, reuse_role,
                                                                        src1_handle, ne10, total_src1_rows, ne10_padded,
                                                                        q8_1_row_size, required_size, layout);
    if (reuse_q8) {
        GGML_SYCL_DEBUG("[MOE-Q8-ARTIFACT] HIT role=%d layer=%d src=%p rows=%lld ne0=%lld bytes=%zu q8=%p\n",
                        (int) reuse_role, reuse_layer, (const void *) src1_d, (long long) total_src1_rows,
                        (long long) ne10, required_size, g_mxfp4_moe_tg_reuse.q8_alloc.ptr);
    }

    ggml_sycl_pool_alloc<int8_t> src1_q8_1_pool(ctx.pool());
    void *                       q8_1_buffer = nullptr;
    if (reuse_candidate) {
        q8_1_buffer = mxfp4_moe_tg_reuse_get_or_alloc_q8(stream, runtime_device, required_size);
    }
    if (!q8_1_buffer) {
        src1_q8_1_pool.alloc(required_size);
        q8_1_buffer = src1_q8_1_pool.get();
        if (reuse_candidate) {
            mxfp4_moe_tg_reuse_invalidate();
        }
    }

    auto t_quant_begin = std::chrono::high_resolution_clock::now();
    if (!reuse_q8) {
        if (tg_profile) {
            stream->wait();
            t_quant_begin = std::chrono::high_resolution_clock::now();
        }
        if (y_soa) {
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows,
                                                                  ne10_padded, stream);
        } else {
            quantize_row_q8_1_sycl<quantize_q8_1>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows, ne10_padded,
                                                  stream);
        }
        if (tg_profile) {
            stream->wait();
        }
    } else if (tg_profile) {
        t_quant_begin = std::chrono::high_resolution_clock::now();
    }
    auto t_quant_end = std::chrono::high_resolution_clock::now();
    if (reuse_candidate && !reuse_q8 && q8_1_buffer == g_mxfp4_moe_tg_reuse.q8_alloc.ptr) {
        mxfp4_moe_tg_reuse_store(runtime_device, reuse_layer, reuse_role, src1_handle, ne10, total_src1_rows,
                                 ne10_padded, q8_1_row_size, required_size, layout);
    } else if (reuse_q8) {
        mxfp4_moe_tg_reuse_invalidate();
    } else if (!reuse_candidate) {
        mxfp4_moe_tg_reuse_invalidate();
    }

    // --- Build routing payload for batched dispatch ---
    // When the selected GPU entries cover the whole (token, top-k) grid and
    // the caller provides a device ids tensor, derive a compact per-row pointer
    // payload on device.  The source table is still descriptor/unified-cache
    // owned; the compact list is only the kernel ABI payload.  With ids=nullptr
    // kernels index expert_ptrs[batch_idx] directly and avoid the expert-id
    // table lookup in the hot loop.
    const int64_t num_tokens    = ne12;
    const int64_t total_batches = n_ids * num_tokens;

    // Q8_1 strides (matching ggml_sycl_mul_mat_id_vec_q)
    const int64_t q8_nb11 = q8_1_row_size;
    const int64_t q8_nb12 = ne11 * q8_1_row_size;

    // stride between expert weight matrices
    const int64_t stride_expert_x = nb02;

    const bool full_gpu_cover = n_gpu_entries == total_batches;
    bool       can_use_direct_ids =
        full_gpu_cover && direct_ids_device != nullptr && direct_ids_nb0 > 0 && direct_ids_nb1 > 0;
    if (can_use_direct_ids) {
        constexpr int64_t    seen_stack_size             = 16;
        uint8_t              seen_stack[seen_stack_size] = {};
        std::vector<uint8_t> seen_heap;
        uint8_t *            seen = seen_stack;
        if (total_batches > seen_stack_size) {
            seen_heap.assign(static_cast<size_t>(total_batches), 0);
            seen = seen_heap.data();
        }
        for (int i = 0; i < n_gpu_entries; i++) {
            const int64_t slot_idx = gpu_ids[i] * num_tokens + gpu_iid1s[i];
            if (slot_idx < 0 || slot_idx >= total_batches || seen[slot_idx]) {
                can_use_direct_ids = false;
                break;
            }
            seen[slot_idx] = true;
        }
    }

    auto t_batch_begin = std::chrono::high_resolution_clock::now();
    if (tg_profile) {
        stream->wait();
        t_batch_begin = std::chrono::high_resolution_clock::now();
    }
    int32_t * ids_device = const_cast<int32_t *>(direct_ids_device);
    if (!can_use_direct_ids) {
        // Build full-size expert_id array for all slots.  GPU entries get
        // their real expert_id; CPU entries get the first GPU entry's
        // expert_id as sentinel (output is overwritten by CPU memcpy).
        const int32_t sentinel_id = gpu_expert_ids[0];

        // Use thread-local storage for batch_ids to prevent use-after-free.
        // The H2D memcpy is async on the in-order queue — the DMA may be deferred
        // behind prior commands (Q8_1 quantization kernel).  If batch_ids is a local
        // std::vector, it's destroyed when this function returns.  But the function
        // returns BEFORE the DMA completes (the kernel and DMA are in-flight on the
        // GPU).  The DMA then reads freed heap memory → stale expert IDs → GPU page
        // fault → DEVICE_LOST.
        //
        // Thread-local storage persists until the next call to this function, which
        // can only happen after the in-order queue has completed all prior commands
        // (including the DMA and kernel from this call).  This guarantees the source
        // buffer is alive for the entire DMA duration without any explicit wait.
        static thread_local std::vector<int32_t> batch_ids;
        batch_ids.assign(total_batches, sentinel_id);
        for (int i = 0; i < n_gpu_entries; i++) {
            const int64_t slot_idx = gpu_ids[i] * num_tokens + gpu_iid1s[i];
            if (slot_idx >= 0 && slot_idx < total_batches) {
                batch_ids[slot_idx] = gpu_expert_ids[i];
            }
        }

        // Allocate device buffer for batch ids and upload.
        ggml_sycl_pool_alloc<int32_t> ids_pool(ctx.pool());
        ids_pool.alloc(total_batches * sizeof(int32_t));
        ids_device = ids_pool.get();
        stream->memcpy(ids_device, batch_ids.data(), total_batches * sizeof(int32_t));
    }
    if (tg_profile) {
        stream->wait();
    }

    const void * const * dispatch_ptrs = expert_ptrs_device;
    const int32_t *      dispatch_ids  = ids_device;
    int64_t              ids_nb0       = can_use_direct_ids ? direct_ids_nb0 : static_cast<int64_t>(sizeof(int32_t));
    int64_t              ids_nb1 = can_use_direct_ids ? direct_ids_nb1 : n_ids * static_cast<int64_t>(sizeof(int32_t));

    if (can_use_direct_ids) {
        // Compact storage is planner/graph-preload-owned.  The hot dispatch path
        // may consume an existing compact ABI buffer, but should not allocate one.
        constexpr bool allow_compact_alloc = false;
        const bool compact_ready = ggml_sycl_moe_prepare_compact_list(ctx, src0, total_batches, allow_compact_alloc);
        if (compact_ready) {
            auto *  extra_mut = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
            void ** compact_ptrs =
                extra_mut ? static_cast<void **>(extra_mut->moe_expert_ptrs_compact_device[ctx.device]) : nullptr;
            if (compact_ptrs) {
                // No missing-flag readback here: this path is only reached when
                // the caller already resolved every selected GPU entry.
                ggml_sycl_build_moe_compact_list(*stream, compact_ptrs, expert_ptrs_device, ids_device, n_ids,
                                                 num_tokens, n_experts, ids_nb0, ids_nb1, nullptr, {});
                dispatch_ptrs = compact_ptrs;
                dispatch_ids  = nullptr;
                ids_nb0       = 0;
                ids_nb1       = 0;
                static std::atomic<int> compact_log{ 0 };
                const char *            row_agg_debug = std::getenv("GGML_SYCL_MOE_ROW_AGG_DEBUG");
                if (row_agg_debug && std::atoi(row_agg_debug) != 0 &&
                    compact_log.fetch_add(1, std::memory_order_relaxed) < 32) {
                    fprintf(stderr,
                            "[MOE-ROW-AGG] stage=compact path=mmvq_compact tensor=%s layout=%d "
                            "entries=%d total_batches=%lld topk=%lld tokens=%lld device=%d\n",
                            src0->name ? src0->name : "?", (int) layout, n_gpu_entries, (long long) total_batches,
                            (long long) n_ids, (long long) num_tokens, ctx.device);
                }
            }
        }
    }

    auto t_batch_end = std::chrono::high_resolution_clock::now();

    auto t_kernel_begin = std::chrono::high_resolution_clock::now();
    if (tg_profile) {
        stream->wait();
        t_kernel_begin = std::chrono::high_resolution_clock::now();
    }
    sycl::event kernel_event;
    bool        have_kernel_event = false;
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_id_sycl(nullptr,           // vx (unused with expert_ptrs)
                                          dispatch_ptrs,     // expert pointer table or compact row list
                                          q8_1_buffer, dst_d, dispatch_ids,
                                          ne00,              // ncols
                                          ne01,              // nrows_per_expert
                                          total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0,
                                          ids_nb1,           // ids strides
                                          q8_nb11, q8_nb12,  // Q8_1 strides
                                          nb1, nb2,          // dst strides
                                          stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_id_sycl(nullptr, dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids, ne00, ne01,
                                          total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0, ids_nb1,
                                          q8_nb11, q8_nb12, nb1, nb2, stream);
            break;
        case GGML_TYPE_MXFP4:
            if (layout == GGML_LAYOUT_COALESCED) {
                coalesced_mul_mat_vec_mxfp4_q8_1_id_sycl(nullptr,           // vx (unused with expert_ptrs)
                                                         dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                                         ne00,              // ncols (MXFP4)
                                                         ne10,              // ncols_y (Q8_1 row size for SoA ds offset)
                                                         ne01,              // nrows_per_expert
                                                         ne02,              // num_experts
                                                         total_batches, n_ids, num_tokens, ne11, ids_nb0,
                                                         ids_nb1,           // ids strides
                                                         q8_nb11, q8_nb12,  // Q8_1 strides
                                                         nb1, nb2,          // dst strides
                                                         stream);
            } else if (layout == GGML_LAYOUT_SOA) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl(nullptr,           // vx (unused with expert_ptrs)
                                                       dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                                       ne00,              // ncols (MXFP4)
                                                       ne10,              // ncols_y (Q8_1 row size for SoA ds offset)
                                                       ne01,              // nrows_per_expert
                                                       ne02,              // num_experts
                                                       total_batches, n_ids, num_tokens, ne11, ids_nb0,
                                                       ids_nb1,           // ids strides
                                                       q8_nb11, q8_nb12,  // Q8_1 strides
                                                       nb1, nb2,          // dst strides
                                                       stream, &kernel_event);
                have_kernel_event = true;
            } else {
                // AOS layout
                mul_mat_vec_mxfp4_q8_1_id_sycl(nullptr,           // vx (unused with expert_ptrs)
                                               dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                               ne00,              // ncols
                                               ne01,              // nrows_per_expert
                                               total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0,
                                               ids_nb1,           // ids strides
                                               q8_nb11, q8_nb12,  // Q8_1 strides
                                               nb1, nb2,          // dst strides
                                               stream);
            }
            break;
        default:
            return false;
    }
    if (tg_profile) {
        if (have_kernel_event) {
            kernel_event.wait();
        } else {
            stream->wait();
        }
    }
    auto t_kernel_end = std::chrono::high_resolution_clock::now();

    auto us = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    const double kernel_wall_us    = us(t_kernel_begin, t_kernel_end);
    const double kernel_event_us   = have_kernel_event ? mmvq_sycl_event_duration_us(kernel_event) : -1.0;
    const double kernel_profile_us = kernel_event_us >= 0.0 ? kernel_event_us : kernel_wall_us;
    g_mmvq_moe_dispatch_timing.activation_quant_us += us(t_quant_begin, t_quant_end);
    g_mmvq_moe_dispatch_timing.batch_id_us += us(t_batch_begin, t_batch_end);
    g_mmvq_moe_dispatch_timing.kernel_submit_us += kernel_wall_us;
    g_mmvq_moe_dispatch_timing.calls++;
    g_mmvq_moe_dispatch_timing.entries += n_gpu_entries;
    if (tg_profile) {
        const mmvq_moe_tg_profile_kind profile_kind =
            reuse_role == mxfp4_moe_role::DOWN ? mmvq_moe_tg_profile_kind::DOWN : mmvq_moe_tg_profile_kind::OTHER;
        mmvq_moe_tg_profile_record(layout, n_gpu_entries, total_batches, us(t_quant_begin, t_quant_end),
                                   us(t_batch_begin, t_batch_end), kernel_profile_us, 0.0, profile_kind);
    }

    GGML_SYCL_DEBUG(
        "[MOE-BATCHED] Dispatched %d GPU experts in single kernel "
        "(type=%d, total_batches=%lld)\n",
        n_gpu_entries, src0->type, (long long) total_batches);
    return true;
}

bool mmvq_moe_batched_dispatch_pair_mxfp4_soa(ggml_backend_sycl_context & ctx,
                                              const ggml_tensor *         src0_a,
                                              const ggml_tensor *         src0_b,
                                              const ggml_tensor *         src1,
                                              ggml_tensor *               dst_a,
                                              ggml_tensor *               dst_b,
                                              const void * const *        expert_ptrs_a_device,
                                              const void * const *        expert_ptrs_b_device,
                                              const int32_t *             ids_device,
                                              int                         n_gpu_entries,
                                              int64_t                     n_ids,
                                              int64_t                     ids_nb0,
                                              int64_t                     ids_nb1) {
    if (n_gpu_entries <= 0 || !expert_ptrs_a_device || !expert_ptrs_b_device || !ids_device) {
        return false;
    }
    if (!src0_a || !src0_b || !src1 || !dst_a || !dst_b || src0_a->type != GGML_TYPE_MXFP4 ||
        src0_b->type != GGML_TYPE_MXFP4) {
        return false;
    }

    const ggml_tensor * src0 = src0_a;
    ggml_tensor *       dst  = dst_a;
    GGML_TENSOR_BINARY_OP_LOCALS;
    if (ne12 != 1 || src0_b->ne[0] != ne00 || src0_b->ne[1] != ne01 || src0_b->ne[2] != ne02) {
        return false;
    }

    const queue_ptr stream  = ctx.stream();
    const float *   src1_d  = static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(src1, ctx.device));
    float *         dst_a_d = static_cast<float *>(ggml_sycl_resolve_tensor_ptr(dst_a, ctx.device));
    float *         dst_b_d = static_cast<float *>(ggml_sycl_resolve_tensor_ptr(dst_b, ctx.device));
    if (!src1_d || !dst_a_d || !dst_b_d) {
        return false;
    }

    const int64_t ne10_padded     = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size   = ne10_padded * sizeof(block_q8_1) / QK8_1;
    const int64_t total_src1_rows = ne11 * ne12;
    const size_t  required_size   = total_src1_rows * q8_1_row_size;

    ggml_sycl_pool_alloc<int8_t> src1_q8_1_pool(ctx.pool());
    void *                       q8_1_buffer = mxfp4_moe_tg_reuse_get_or_alloc_q8(stream, ctx.device, required_size);
    if (!q8_1_buffer) {
        src1_q8_1_pool.alloc(required_size);
        q8_1_buffer = src1_q8_1_pool.get();
        mxfp4_moe_tg_reuse_invalidate();
    }

    const bool tg_profile    = mmvq_moe_tg_profile_enabled() && src0_a->type == GGML_TYPE_MXFP4 && ne12 == 1;
    auto       t_quant_begin = std::chrono::high_resolution_clock::now();
    if (tg_profile) {
        stream->wait();
        t_quant_begin = std::chrono::high_resolution_clock::now();
    }
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows,
                                                          ne10_padded, stream);
    if (tg_profile) {
        stream->wait();
    }
    auto t_quant_end = std::chrono::high_resolution_clock::now();

    const int64_t num_tokens    = ne12;
    const int64_t total_batches = n_ids * num_tokens;
    auto          t_batch_begin = std::chrono::high_resolution_clock::now();
    auto          t_batch_end   = std::chrono::high_resolution_clock::now();
    const int64_t q8_nb11       = q8_1_row_size;
    const int64_t q8_nb12       = ne11 * q8_1_row_size;

    auto t_kernel_begin = std::chrono::high_resolution_clock::now();
    reorder_mul_mat_vec_mxfp4_q8_1_id_pair_sycl(expert_ptrs_a_device, expert_ptrs_b_device, q8_1_buffer, dst_a_d,
                                                dst_b_d, ids_device, ne00, ne10, ne01, total_batches, n_ids, num_tokens,
                                                ne11, ids_nb0, ids_nb1, q8_nb11, q8_nb12, nb1, nb2, dst_b->nb[1],
                                                dst_b->nb[2], stream);
    if (tg_profile) {
        stream->wait();
    }
    auto t_kernel_end = std::chrono::high_resolution_clock::now();

    auto us = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    const double quant_us  = us(t_quant_begin, t_quant_end);
    const double batch_us  = us(t_batch_begin, t_batch_end);
    const double kernel_us = us(t_kernel_begin, t_kernel_end);
    g_mmvq_moe_dispatch_timing.activation_quant_us += quant_us;
    g_mmvq_moe_dispatch_timing.batch_id_us += batch_us;
    g_mmvq_moe_dispatch_timing.kernel_submit_us += kernel_us;
    g_mmvq_moe_dispatch_timing.calls++;
    g_mmvq_moe_dispatch_timing.entries += 2 * n_gpu_entries;
    if (tg_profile) {
        mmvq_moe_tg_profile_record(GGML_LAYOUT_SOA, 2 * n_gpu_entries, 2 * total_batches, quant_us, batch_us, kernel_us,
                                   0.0, mmvq_moe_tg_profile_kind::OTHER, 2);
    }
    GGML_SYCL_DEBUG("[MOE-PAIR] Dispatched %d GPU experts for two MXFP4 SOA tensors (total_batches=%lld)\n",
                    n_gpu_entries, (long long) total_batches);
    return true;
}

bool mmvq_moe_batched_dispatch_pair_glu_mxfp4_soa(ggml_backend_sycl_context &   ctx,
                                                  const ggml_tensor *           gate_weight,
                                                  const ggml_tensor *           up_weight,
                                                  const ggml_tensor *           src1,
                                                  ggml_tensor *                 glu_dst,
                                                  const void * const *          gate_ptrs_device,
                                                  const void * const *          up_ptrs_device,
                                                  const int32_t *               ids_device,
                                                  const float *                 gate_bias_device,
                                                  const float *                 up_bias_device,
                                                  int64_t                       gate_bias_nb1,
                                                  int64_t                       up_bias_nb1,
                                                  int                           n_gpu_entries,
                                                  int64_t                       n_ids,
                                                  int64_t                       ids_nb0,
                                                  int64_t                       ids_nb1,
                                                  int                           glu_op,
                                                  float                         alpha,
                                                  float                         limit,
                                                  const ggml_sycl::mem_handle * glu_dst_handle_override,
                                                  bool                          direct_xmx_eligible) {
    if (n_gpu_entries <= 0 || !gate_ptrs_device || !up_ptrs_device || !ids_device) {
        return false;
    }
    if (!gate_weight || !up_weight || !src1 || !glu_dst || gate_weight->type != GGML_TYPE_MXFP4 ||
        up_weight->type != GGML_TYPE_MXFP4 || glu_dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_SWIGLU_OAI) {
        return false;
    }

    const ggml_tensor * src0 = gate_weight;
    ggml_tensor *       dst  = glu_dst;
    GGML_TENSOR_BINARY_OP_LOCALS;
    if (ne12 != 1 || up_weight->ne[0] != ne00 || up_weight->ne[1] != ne01 || up_weight->ne[2] != ne02 ||
        glu_dst->ne[0] != ne01) {
        return false;
    }

    const queue_ptr stream = ctx.stream();
    const float *   src1_d = static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(src1, ctx.device));
    float *         glu_d  = static_cast<float *>(ggml_sycl_resolve_tensor_ptr(glu_dst, ctx.device));
    if (!src1_d || !glu_d) {
        return false;
    }

    const int64_t ne10_padded     = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size   = ne10_padded * sizeof(block_q8_1) / QK8_1;
    const int64_t total_src1_rows = ne11 * ne12;
    const size_t  required_size   = total_src1_rows * q8_1_row_size;

    ggml_sycl_pool_alloc<int8_t> src1_q8_1_pool(ctx.pool());
    void *                       q8_1_buffer = mxfp4_moe_tg_reuse_get_or_alloc_q8(stream, ctx.device, required_size);
    if (!q8_1_buffer) {
        src1_q8_1_pool.alloc(required_size);
        q8_1_buffer = src1_q8_1_pool.get();
        mxfp4_moe_tg_reuse_invalidate();
    }

    const bool tg_profile    = mmvq_moe_tg_profile_enabled() && ne12 == 1;
    auto       t_quant_begin = std::chrono::high_resolution_clock::now();
    if (tg_profile) {
        stream->wait();
        t_quant_begin = std::chrono::high_resolution_clock::now();
    }
    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows,
                                                          ne10_padded, stream);
    if (tg_profile) {
        stream->wait();
    }
    auto t_quant_end = std::chrono::high_resolution_clock::now();

    const int64_t num_tokens    = ne12;
    const int64_t total_batches = n_ids * num_tokens;
    const int64_t q8_nb11       = q8_1_row_size;
    const int64_t q8_nb12       = ne11 * q8_1_row_size;

    auto        t_kernel_begin  = std::chrono::high_resolution_clock::now();
    bool        used_direct_xmx = false;
    sycl::event kernel_event;
    bool        have_kernel_event = false;
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    const auto & xmx_caps = ggml_sycl_info().devices[ctx.device].xmx_caps;
    if (mmvq_moe_direct_xmx_glu_enabled() && direct_xmx_eligible && xmx_caps.supported && xmx_caps.supports_int8) {
        reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl(
            gate_ptrs_device, up_ptrs_device, q8_1_buffer, glu_d, ids_device, gate_bias_device, up_bias_device, ne00,
            ne10, ne01, total_batches, n_ids, num_tokens, ne11, ids_nb0, ids_nb1, q8_nb11, q8_nb12, glu_dst->nb[1],
            glu_dst->nb[2], gate_bias_nb1, up_bias_nb1, glu_op, alpha, limit, stream);
        used_direct_xmx = true;
    }
#endif
    if (!used_direct_xmx) {
        reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl(
            gate_ptrs_device, up_ptrs_device, q8_1_buffer, glu_d, ids_device, gate_bias_device, up_bias_device, ne00,
            ne10, ne01, total_batches, n_ids, num_tokens, ne11, ids_nb0, ids_nb1, q8_nb11, q8_nb12, glu_dst->nb[1],
            glu_dst->nb[2], gate_bias_nb1, up_bias_nb1, glu_op, alpha, limit, stream, &kernel_event);
        have_kernel_event = true;
    }
    if (tg_profile) {
        if (have_kernel_event) {
            kernel_event.wait();
        } else {
            stream->wait();
        }
    }
    auto t_kernel_end = std::chrono::high_resolution_clock::now();

    auto us = [](std::chrono::high_resolution_clock::time_point a, std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    double        quant_us           = us(t_quant_begin, t_quant_end);
    double        down_q8_publish_us = 0.0;
    const int     down_layer         = mxfp4_moe_layer_from_name(gate_weight->name);
    const int64_t glu_rows           = glu_dst->ne[1] * glu_dst->ne[2];
    if (glu_dst->nb[0] == sizeof(float) && glu_dst->nb[1] == static_cast<size_t>(glu_dst->ne[0]) * sizeof(float)) {
        if (mxfp4_moe_tg_publish_q8_soa(stream, ctx.device, down_layer, mxfp4_moe_role::DOWN, glu_d, glu_dst->ne[0],
                                        glu_rows, &down_q8_publish_us, glu_dst_handle_override)) {
            quant_us += down_q8_publish_us;
        }
    }
    const double kernel_wall_us    = us(t_kernel_begin, t_kernel_end);
    const double kernel_event_us   = have_kernel_event ? mmvq_sycl_event_duration_us(kernel_event) : -1.0;
    const double kernel_profile_us = kernel_event_us >= 0.0 ? kernel_event_us : kernel_wall_us;
    g_mmvq_moe_dispatch_timing.activation_quant_us += quant_us;
    g_mmvq_moe_dispatch_timing.kernel_submit_us += kernel_wall_us;
    g_mmvq_moe_dispatch_timing.calls++;
    g_mmvq_moe_dispatch_timing.entries += 2 * n_gpu_entries;
    if (tg_profile) {
        mmvq_moe_tg_profile_record(GGML_LAYOUT_SOA, 2 * n_gpu_entries, 2 * total_batches, quant_us - down_q8_publish_us,
                                   0.0, kernel_profile_us, down_q8_publish_us, mmvq_moe_tg_profile_kind::GATEUP_GLU, 2);
    }
    GGML_SYCL_DEBUG("[MOE-PAIR-GLU] Dispatched %d GPU experts for fused MXFP4 SOA gate/up/GLU\n", n_gpu_entries);
    return true;
}

bool ggml_sycl_mul_mat_id_vec_q(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                const ggml_tensor *         ids,
                                ggml_tensor *               dst,
                                const layout_mode *         forced_layout) {
    GGML_SYCL_DEBUG("[MMVQ-ENTRY] src0=%s type=%d forced_layout=%d\n", src0->name ? src0->name : "?", src0->type,
                    forced_layout ? (int) *forced_layout : -1);
    // Early batch size check — route large PP batches to the per-expert batched
    // GEMM path (host-side routing → oneDNN/MMQ) instead of MMVQ vec_dot.
    //
    // Trade-off:
    //   MMVQ: ONE kernel launch dispatching all (token, expert) pairs in parallel.
    //         Each work-group does an independent vec_dot — no weight reuse across
    //         tokens assigned to the same expert.
    //   PP fallback: iterates active experts, gathers tokens per expert, calls
    //         ggml_sycl_mul_mat with batched rows.  When batch >= oneDNN threshold
    //         (default 16), this uses XMX GEMM with full weight reuse.  Below that
    //         threshold it falls to MMVQ internally, adding gather/scatter overhead.
    //
    // Adaptive threshold: switch when the average tokens per expert is large enough
    // for oneDNN to provide a benefit.  avg_tokens_per_expert = batch * top_k / n_experts.
    // We want avg >= oneDNN PP threshold (default 16) for the batched path to win.
    // Below that, ggml_sycl_mul_mat falls back to MMVQ internally, adding
    // per-expert gather/scatter overhead with no GEMM benefit.
    // Rearranging: batch >= n_experts * onednn_threshold / top_k.
    //
    // Examples (onednn_threshold=16):
    //   128 experts, top-4 → threshold = 512  (PP512+ uses batched GEMM)
    //     8 experts, top-2 → threshold =  64  (PP64+  uses batched GEMM)
    //
    // Override: GGML_SYCL_MMVQ_MOE_PP=N forces a fixed threshold (batch > N → batched).
    //           N=0: always batched; N=9999: always MMVQ.
    static int64_t MMVQ_MOE_MAX_BATCH_OVERRIDE = []() -> int64_t {
        const char * env = std::getenv("GGML_SYCL_MMVQ_MOE_PP");
        return env ? std::atoi(env) : -1;    // -1 = use adaptive
    }();
    const int64_t batch_size = src1->ne[2];  // ne12 - number of tokens
    {
        int64_t effective_threshold;
        if (MMVQ_MOE_MAX_BATCH_OVERRIDE >= 0) {
            effective_threshold = MMVQ_MOE_MAX_BATCH_OVERRIDE;
        } else {
            // Adaptive: switch when avg tokens/expert >= oneDNN PP threshold.
            // Read the threshold once (matches the value in ggml_sycl_mul_mat).
            static int64_t onednn_pp_thr = []() -> int64_t {
                const char * env = std::getenv("GGML_SYCL_ONEDNN_PP_MIN_BATCH");
                int          v   = env ? std::atoi(env) : 16;
                return v > 0 ? v : 16;
            }();
            const int64_t n_experts = src0->ne[2];
            const int64_t top_k     = ids->ne[0];
            effective_threshold     = (top_k > 0) ? (n_experts * onednn_pp_thr / top_k) : 1;
            if (effective_threshold < 1) {
                effective_threshold = 1;
            }
        }
        // Log the computed threshold once per model configuration
        {
            static std::atomic<int> threshold_logged{ 0 };
            if (threshold_logged.fetch_add(1, std::memory_order_relaxed) == 0) {
                GGML_LOG_INFO("[MMVQ-MoE] PP batch threshold = %ld (n_experts=%ld top_k=%ld%s)\n",
                              (long) effective_threshold, (long) src0->ne[2], (long) ids->ne[0],
                              MMVQ_MOE_MAX_BATCH_OVERRIDE >= 0 ? " override" : " adaptive");
            }
        }
        if (batch_size > effective_threshold) {
            GGML_SYCL_DEBUG("[MMVQ] Batch %ld > threshold %ld, routing to batched GEMM path\n", (long) batch_size,
                            (long) effective_threshold);
            return false;
        }
    }
    const auto *      src0_extra   = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    // Detect host-resident weights including SYCL HOST_PINNED buffers
    bool              host_weights = ggml_sycl_is_host_resident_weight(src0, ctx.stream());
    const layout_mode layout =
        forced_layout ? *forced_layout : ggml_sycl_select_moe_mmvq_layout(src0, ctx.device, host_weights);
    // For MXFP4 MoE: allow SOA/Coalesced dispatch even when base tensor is AOS.
    // The expert pointer table + unified cache will stage per-expert data in the
    // requested reordered layout on-the-fly. This avoids the slow AOS kernel.
    const bool mxfp4_moe_reorder_dispatch = (src0->type == GGML_TYPE_MXFP4) &&
                                            (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED) &&
                                            src0_extra && (get_effective_layout_mode(src0_extra) == GGML_LAYOUT_AOS);

    // Always validate that the forced layout is actually supported by the tensor
    // dimensions.  COALESCED requires blocks_per_row % 32 == 0 — reject if not met,
    // even for mxfp4_moe_reorder_dispatch (the cache cannot produce a valid coalesced
    // reorder for non-aligned tensors).
    if (forced_layout) {
        const layout_mode resolved = ggml_sycl_adjust_layout_for_tensor(src0, *forced_layout, ctx.device);
        if (resolved != *forced_layout) {
            GGML_SYCL_DEBUG("[MMVQ] Layout=%d unsupported for %s (resolved=%d)\n", (int) *forced_layout, src0->name,
                            (int) resolved);
            return false;
        }
    }
    if (!host_weights && src0_extra && !mxfp4_moe_reorder_dispatch) {
        const layout_mode effective          = get_effective_layout_mode(src0_extra);
        const bool        allow_aos_fallback = forced_layout && layout == GGML_LAYOUT_AOS;
        if (effective != layout && !allow_aos_fallback) {
            GGML_SYCL_DEBUG("[MMVQ] Layout=%d mismatches effective=%d for %s\n", (int) layout, (int) effective,
                            src0->name ? src0->name : "?");
            return false;
        }
        if (effective != layout && allow_aos_fallback) {
            GGML_SYCL_DEBUG("[MMVQ] Using AoS fallback for %s (effective=%d)\n", src0->name ? src0->name : "?",
                            (int) effective);
        }
    }
    if ((src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q8_0) && layout != GGML_LAYOUT_AOS) {
        GGML_SYCL_DEBUG("[MMVQ] Layout=%d unsupported for MoE type=%d (%s)\n", (int) layout, src0->type,
                        src0->name ? src0->name : "?");
        return false;
    }
    // MXFP4 MoE supports SoA/Coalesced layouts via unified cache staging + device ptr tables.
    // Enable ptr_table when host weights need staging OR when MoE reorder dispatch
    // needs the expert cache to produce reordered per-expert data from AOS base tensor.
    const bool use_ptr_table = host_weights || mxfp4_moe_reorder_dispatch;
    if (mxfp4_moe_reorder_dispatch) {
        GGML_SYCL_DEBUG("[MMVQ] MXFP4 MoE reorder dispatch: layout=%d use_ptr_table=1 for %s\n", (int) layout,
                        src0->name ? src0->name : "?");
    }
    // XMX tiled layout is handled by XMX paths, not MMVQ
    if (layout == GGML_LAYOUT_XMX_TILED) {
        GGML_SYCL_DEBUG("[MMVQ] XMX tiled layout for %s, skipping MMVQ\n", src0->name);
        return false;
    }

    // Batch size check already performed at function entry (adaptive threshold).

    GGML_TENSOR_BINARY_OP_LOCALS;

    // Supports both ne12 == 1 (decode) and ne12 > 1 (prompt, up to threshold)
    // The kernel dispatches over (iid1, id) pairs in parallel

    // Only handle quantized types that have _id kernels
    if (!ggml_is_quantized(src0->type)) {
        return false;
    }

    // Check for supported types
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return false;  // Fall back to host routing for other types
    }

    const queue_ptr stream                = ctx.stream();
    auto *          route_cache           = ggml_sycl::get_unified_cache(*stream);
    const bool      placement_plan_active = route_cache && route_cache->has_placement_plan();
    // Previously returned false here when (placement_plan_active && use_ptr_table).
    // The mixed-ptr check at ~3870-3886 is the architecturally correct gate: it
    // inspects actual expert pointer residency after ggml_sycl_update_moe_ptr_table
    // builds the table (with skip_cpu_routed_experts already honoring the plan),
    // and falls back only when pointers are actually mixed — not merely because
    // a plan exists.  Blanket-disabling MMVQ under a plan forced every 20B MoE
    // MUL_MAT_ID through the hybrid dispatch_cpu_compute path, which had
    // task-construction bugs (l144i diagnostic) producing garbled output.

    // Quantize src1 to Q8_1 format - need to handle all rows (ne11 * ne12)
    const int64_t ne10_padded   = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;

    // Total rows = ne11 * ne12 (e.g., 4 expert outputs * 1 token = 4 rows)
    const int64_t total_src1_rows = ne11 * ne12;
    const size_t  required_size   = total_src1_rows * q8_1_row_size;

    const float *   src1_d = static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(src1, ctx.device));
    sycl::event     ids_copy_event;
    int64_t         ids_nb0 = ids->nb[0];
    int64_t         ids_nb1 = ids->nb[1];
    const int32_t * ids_d   = ggml_sycl_get_moe_ids_device_ptr(ctx, ids, &ids_copy_event, &ids_nb0, &ids_nb1);
    if (!ids_d) {
        GGML_SYCL_DEBUG("[MMVQ] Missing ids device pointer for %s\n", src0->name);
        return false;
    }
    float * dst_d = static_cast<float *>(ggml_sycl_resolve_tensor_ptr(dst, ctx.device));
    if (!src1_d || !dst_d) {
        GGML_SYCL_DEBUG("[MMVQ] Missing resolved ptrs for MoE _id dispatch (%s)\n", src0->name);
        return false;
    }

    const void * const * expert_ptrs = nullptr;
    sycl::event          table_event;
    bool                 has_table_event = false;
    if (use_ptr_table) {
        if (g_ggml_sycl_graph_recording) {
            if (src0_extra && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
                expert_ptrs = static_cast<const void * const *>(src0_extra->moe_expert_ptrs_device[ctx.device]);
            }
            if (!expert_ptrs) {
                GGML_SYCL_DEBUG("[MMVQ] Missing expert pointer table during graph recording for %s\n", src0->name);
                ctx.moe_graphs_disabled_once = true;
                return false;
            }
            if (src0_extra) {
                const auto & host_ptrs = src0_extra->moe_expert_ptrs_host[ctx.device];
                if (host_ptrs.empty()) {
                    GGML_SYCL_DEBUG("[MMVQ] Empty expert pointer table during graph recording for %s\n", src0->name);
                    ctx.moe_graphs_disabled_once = true;
                    return false;
                }
                table_event     = stream->memcpy(src0_extra->moe_expert_ptrs_device[ctx.device], host_ptrs.data(),
                                                 host_ptrs.size() * sizeof(void *));
                has_table_event = true;
            }
        } else {
            const bool allow_all_experts = false;
            // force_cache_aos=true ensures experts are staged to GPU memory even for AoS layout
            // This is critical for mmap'd weights which cannot be accessed directly by GPU kernels
            GGML_SYCL_DEBUG("[MMVQ] About to call ggml_sycl_update_moe_ptr_table layout=%d\n", (int) layout);
            if (!ggml_sycl_update_moe_ptr_table(ctx, src0, ids, layout, &table_event, allow_all_experts, nullptr,
                                                /*skip_device_copy=*/false,
                                                /*force_cache_aos=*/host_weights,
                                                /*skip_cpu_routed_experts=*/host_weights || placement_plan_active)) {
                GGML_SYCL_DEBUG("[MMVQ] Failed to update expert pointer table for %s\n", src0->name);
                return false;
            }
            GGML_SYCL_DEBUG("[MMVQ] ggml_sycl_update_moe_ptr_table succeeded\n");
            has_table_event = true;
            if (src0_extra && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
                expert_ptrs = static_cast<const void * const *>(src0_extra->moe_expert_ptrs_device[ctx.device]);
                // T4 Gap 2a: previously gated on placement_plan_active only; the
                // planner-inactive + host_weights + mixed-ptrs triangle (e.g. VRAM
                // budget exhaustion with mmap'd MXFP4 experts) also needs to bail
                // so the caller can retry with CPU dispatch. Fire whenever the
                // pointer table is mixed, regardless of planner state.
                if (placement_plan_active || host_weights) {
                    const auto & host_ptrs  = src0_extra->moe_expert_ptrs_host[ctx.device];
                    bool         mixed_ptrs = false;
                    for (const void * ptr : host_ptrs) {
                        if (ptr && ggml_sycl_get_alloc_type(ptr) != sycl::usm::alloc::device) {
                            mixed_ptrs = true;
                            break;
                        }
                    }
                    if (mixed_ptrs) {
                        GGML_SYCL_DEBUG(
                            "[MMVQ] Mixed device/host expert pointers for %s (plan=%d host=%d); "
                            "falling back to hybrid dispatch\n",
                            src0->name ? src0->name : "?", placement_plan_active ? 1 : 0, host_weights ? 1 : 0);
                        return false;
                    }
                }
            }
            if (!expert_ptrs) {
                GGML_SYCL_DEBUG("[MMVQ] Missing expert pointer table for %s\n", src0->name);
                return false;
            }
            GGML_SYCL_DEBUG("[MMVQ] expert_ptrs=%p\n", (void *) expert_ptrs);
        }
    }

    const void * src0_d = nullptr;
    if (!use_ptr_table) {
        auto resolved = ggml_sycl_resolve(src0, ctx.device);
        if (!resolved) {
            GGML_SYCL_DEBUG("[MMVQ] resolve_weight failed for %s (layout=%d)\n", src0->name ? src0->name : "?",
                            (int) layout);
            return false;
        }
        src0_d = resolved.ptr;
    }

    // Check Q8_1 quantization cache - MoE uses same input for gate/up/down projections
    void * q8_1_buffer        = nullptr;
    bool   using_cached       = false;
    bool   using_preallocated = false;

#ifdef GGML_SYCL_GRAPH
    // Check cache first - avoids re-quantizing same input across MoE projections
    if (ctx.moe_q8_cache.matches(src1_d, ne10, total_src1_rows)) {
        q8_1_buffer  = ctx.moe_q8_cache.cached_q8_1;
        using_cached = true;
        GGML_SYCL_DEBUG("[MOE-CACHE] Cache HIT: src1=%p, ne10=%lld, rows=%lld\n", src1_d, (long long) ne10,
                        (long long) total_src1_rows);
    }
#endif

    // Fall back to allocation + quantization if cache miss
    ggml_sycl_pool_alloc<int8_t> src1_q8_1_pool(ctx.pool());
    if (!using_cached) {
#ifdef GGML_SYCL_GRAPH
        // Try pre-allocated buffer for graph recording
        if (g_ggml_sycl_graph_recording && ctx.moe_buffers.initialized) {
            q8_1_buffer = ctx.moe_buffers.get_next_buffer(required_size);
            if (q8_1_buffer) {
                using_preallocated = true;
                GGML_SYCL_DEBUG("[MOE-GRAPH] Using pre-allocated buffer %d\n", ctx.moe_buffers.current_buffer_idx - 1);
            }
        }
#endif

        // Fall back to pool allocation if no pre-allocated buffer available
        if (!using_preallocated) {
            src1_q8_1_pool.alloc(required_size);
            q8_1_buffer = src1_q8_1_pool.get();
        }

        // Quantize all rows to Q8_1
        // Use SoA quantizer if weights are in any reordered layout (both SoA and COALESCED kernels expect SoA Y)
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        const bool              y_soa = (layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED);
        if (y_soa) {
            GGML_SYCL_DEBUG("[MoE-Q8_1] Quantizing Y to SoA layout (X is_soa=%d)\n", y_soa);
            quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows,
                                                                  ne10_padded, stream);
        } else {
            GGML_SYCL_DEBUG("[MoE-Q8_1] Quantizing Y to AoS layout (X is_soa=%d)\n", y_soa);
            quantize_row_q8_1_sycl<quantize_q8_1>(src1_d, (char *) q8_1_buffer, ne10, total_src1_rows, ne10_padded,
                                                  stream);
        }

#ifdef GGML_SYCL_GRAPH
        // Cache the quantized result for subsequent gate/up/down calls
        // Only cache if using pre-allocated buffer (pool buffers get freed)
        if (using_preallocated) {
            ctx.moe_q8_cache.cached_q8_1 = q8_1_buffer;
            ctx.moe_q8_cache.cached_src  = src1_d;
            ctx.moe_q8_cache.cached_ne10 = ne10;
            ctx.moe_q8_cache.cached_rows = total_src1_rows;
            ctx.moe_q8_cache.cached_size = required_size;
            ctx.moe_q8_cache.valid       = true;
            GGML_SYCL_DEBUG("[MOE-CACHE] Cache STORE: src1=%p, ne10=%lld, rows=%lld, buffer=%p\n", src1_d,
                            (long long) ne10, (long long) total_src1_rows, q8_1_buffer);
        }
#endif
    }

    // Calculate strides from tensors (matching host-side logic)
    const int64_t n_ids      = ids->ne[0];  // Number of expert selections per token
    const int64_t num_tokens = ids->ne[1];  // Number of tokens (should equal ne12)

    // Total batches = tokens * expert_selections_per_token
    const int64_t total_batches = num_tokens * n_ids;

    // stride_expert_x: offset between expert weight matrices
    const int64_t stride_expert_x = nb02;

    // Quantized src1 strides (map to Q8_1 layout)
    // For Q8_1: each row is q8_1_row_size bytes
    // q8_nb11: stride between consecutive rows (dimension 1)
    // q8_nb12: stride between tokens (dimension 2) = ne11 rows per token
    const int64_t q8_nb11 = q8_1_row_size;
    const int64_t q8_nb12 = ne11 * q8_1_row_size;

    const bool allow_compact_alloc = !g_ggml_sycl_graph_recording;
    const bool compact_ready =
        expert_ptrs && ggml_sycl_moe_prepare_compact_list(ctx, src0, total_batches, allow_compact_alloc);
    const void * const * dispatch_ptrs = expert_ptrs;
    const int32_t *      dispatch_ids  = ids_d;

    if (compact_ready && expert_ptrs) {
        auto *  extra_mut = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        void ** compact_ptrs =
            extra_mut ? static_cast<void **>(extra_mut->moe_expert_ptrs_compact_device[ctx.device]) : nullptr;
        if (compact_ptrs) {
            int *                    missing_device = (!g_ggml_sycl_graph_recording && extra_mut) ?
                                                          extra_mut->moe_expert_ptrs_missing_device[ctx.device] :
                                                          nullptr;
            std::vector<sycl::event> compact_deps;
            if (!g_ggml_sycl_graph_recording && has_table_event) {
                compact_deps.push_back(table_event);
            }
            if (g_ggml_sycl_graph_recording) {
                if (has_table_event) {
                    compact_deps.push_back(table_event);
                }
                if (ids->buffer && ggml_backend_buffer_is_host(ids->buffer)) {
                    compact_deps.push_back(ids_copy_event);
                }
            }
            sycl::event build_event =
                ggml_sycl_build_moe_compact_list(*stream, compact_ptrs, expert_ptrs, ids_d, n_ids, num_tokens, ne02,
                                                 ids_nb0, ids_nb1, missing_device, compact_deps);
            if (!g_ggml_sycl_graph_recording && missing_device) {
                int missing_host = 0;
                stream->memcpy(&missing_host, missing_device, sizeof(int), build_event).wait();
                if (missing_host) {
                    GGML_SYCL_DEBUG("[MMVQ] Missing cached expert pointer(s) for %s, falling back\n", src0->name);
                    ctx.moe_graphs_disabled_once = g_ggml_sycl_graph_recording;
                    return false;
                }
            }
            dispatch_ptrs = compact_ptrs;
            dispatch_ids  = nullptr;
        } else if (g_ggml_sycl_graph_recording) {
            GGML_SYCL_DEBUG("[MMVQ] Missing compact list buffer during graph recording for %s\n", src0->name);
            ctx.moe_graphs_disabled_once = true;
            return false;
        }
    }

    // Dispatch based on type
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_id_sycl(src0_d, dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                          ne00,              // ncols
                                          ne01,              // nrows_per_expert
                                          total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0,
                                          ids_nb1,           // ids strides
                                          q8_nb11, q8_nb12,  // Q8_1 strides
                                          nb1, nb2,          // dst strides
                                          stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_id_sycl(src0_d, dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                          ne00,              // ncols
                                          ne01,              // nrows_per_expert
                                          total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0,
                                          ids_nb1,           // ids strides
                                          q8_nb11, q8_nb12,  // Q8_1 strides
                                          nb1, nb2,          // dst strides
                                          stream);
            break;
        case GGML_TYPE_MXFP4:
            {
                // Check if weights are in SoA layout (CPU reorder sets this during upload)
                auto *     extra          = (ggml_tensor_extra_gpu *) src0->extra;
                const bool x_is_soa       = (layout == GGML_LAYOUT_SOA);
                const bool x_is_coalesced = (layout == GGML_LAYOUT_COALESCED);
                const bool x_is_reordered = x_is_soa || x_is_coalesced;

                GGML_SYCL_DEBUG(
                    "[MMVQ-MXFP4] ENTER: layout=%d use_ptr_table=%d host_weights=%d src0_d=%p dispatch_ptrs=%p\n",
                    (int) layout, use_ptr_table, host_weights, src0_d, (void *) dispatch_ptrs);
                GGML_SYCL_DEBUG("[MMVQ-MXFP4] X: is_soa=%d is_coalesced=%d is_reordered=%d extra=%p tensor=%s\n",
                                x_is_soa, x_is_coalesced, x_is_reordered, extra, src0->name ? src0->name : "null");

                if (x_is_soa || x_is_coalesced) {
                    // SoA layout - both MXFP4 weights and Q8_1 input must be in SoA format
                    // Verify: is_soa should imply is_reordered
                    GGML_ASSERT(x_is_reordered && "X is_soa but not is_reordered - flag inconsistency");

                    // Validate expert pointers: SOA/Coalesced dispatch requires all
                    // active expert pointers to be device-accessible SOA/Coalesced data.
                    // When experts are host-resident and SOA staging was skipped (cache
                    // miss or degradation), the pointer table may contain null entries
                    // or AOS host pointers.  Fall back to AOS dispatch in that case.
                    if (use_ptr_table && host_weights && src0_extra) {
                        const auto & host_ptrs = src0_extra->moe_expert_ptrs_host[ctx.device];
                        bool         ptrs_ok   = true;
                        for (size_t ep = 0; ep < host_ptrs.size() && ptrs_ok; ep++) {
                            if (!host_ptrs[ep]) {
                                continue;  // null = expert not active this token
                            }
                            auto alloc = ggml_sycl_get_alloc_type(host_ptrs[ep]);
                            if (alloc != sycl::usm::alloc::device) {
                                ptrs_ok = false;
                            }
                        }
                        if (!ptrs_ok) {
                            GGML_SYCL_DEBUG(
                                "[MMVQ-MXFP4] SOA dispatch rejected: non-device expert ptrs, falling back\n");
                            return false;  // Fall back to AOS dispatch
                        }
                    }

                    // DEBUG: Verify actual data layout on GPU
                    if (g_ggml_sycl_debug >= 2 && !host_weights) {
                        // For MXFP4 SoA: qs region at offset 0, scales at offset (ncols/2)*total_rows
                        // For Q8_1 SoA: quants at offset 0..ne10-1, ds at offset ne10
                        const int64_t total_x_rows = ne01 * ne02;
                        const int64_t x_qs_size =
                            (ne00 / 2) * total_x_rows;  // 16 bytes per block, ncols/QK_MXFP4 blocks

                        // Read first bytes of X to verify layout
                        uint8_t x_sample[32];
                        stream->memcpy(x_sample, src0_d, 32).wait();
                        fprintf(
                            stderr,
                            "[VERIFY-X] First 16 bytes (should be qs): %02x %02x %02x %02x %02x %02x %02x %02x...\n",
                            x_sample[0], x_sample[1], x_sample[2], x_sample[3], x_sample[4], x_sample[5], x_sample[6],
                            x_sample[7]);

                        // Read scale region (at offset x_qs_size)
                        uint8_t x_scale_sample[8];
                        stream->memcpy(x_scale_sample, (const uint8_t *) src0_d + x_qs_size, 8).wait();
                        fprintf(stderr,
                                "[VERIFY-X] First 8 scales at offset %lld: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                                (long long) x_qs_size, x_scale_sample[0], x_scale_sample[1], x_scale_sample[2],
                                x_scale_sample[3], x_scale_sample[4], x_scale_sample[5], x_scale_sample[6],
                                x_scale_sample[7]);

                        // Read first bytes of Y (Q8_1) to verify layout
                        int8_t y_quant_sample[32];
                        stream->memcpy(y_quant_sample, q8_1_buffer, 32).wait();
                        fprintf(stderr, "[VERIFY-Y] First 32 quants: %d %d %d %d %d %d %d %d...\n", y_quant_sample[0],
                                y_quant_sample[1], y_quant_sample[2], y_quant_sample[3], y_quant_sample[4],
                                y_quant_sample[5], y_quant_sample[6], y_quant_sample[7]);

                        // Read ds values (at offset ne10)
                        sycl::half2 y_ds_sample[4];
                        stream->memcpy(y_ds_sample, (const char *) q8_1_buffer + ne10, sizeof(y_ds_sample)).wait();
                        fprintf(stderr, "[VERIFY-Y] First 4 ds at offset %lld: d0=%.4f s0=%.4f, d1=%.4f s1=%.4f\n",
                                (long long) ne10, (float) y_ds_sample[0][0], (float) y_ds_sample[0][1],
                                (float) y_ds_sample[1][0], (float) y_ds_sample[1][1]);
                    }

                    if (x_is_coalesced) {
                        GGML_SYCL_DEBUG(
                            "[MMVQ-MXFP4-Coalesced] X=Coalesced Y=SoA ne00=%lld ne10=%lld ne01=%lld ne02=%lld\n",
                            (long long) ne00, (long long) ne10, (long long) ne01, (long long) ne02);
                        coalesced_mul_mat_vec_mxfp4_q8_1_id_sycl(src0_d, dispatch_ptrs, q8_1_buffer, dst_d,
                                                                 dispatch_ids,
                                                                 ne00,     // ncols (MXFP4)
                                                                 ne10,     // ncols_y (Q8_1 row size for SoA ds offset)
                                                                 ne01,     // nrows_per_expert
                                                                 ne02,     // num_experts
                                                                 total_batches, n_ids, num_tokens, ne11, ids_nb0,
                                                                 ids_nb1,  // ids strides
                                                                 q8_nb11, q8_nb12,  // Q8_1 strides
                                                                 nb1, nb2,          // dst strides
                                                                 stream);
                    } else {
                        GGML_SYCL_DEBUG("[MMVQ-MXFP4-SoA] X=SoA Y=SoA ne00=%lld ne10=%lld ne01=%lld ne02=%lld\n",
                                        (long long) ne00, (long long) ne10, (long long) ne01, (long long) ne02);

                        // DEBUG: Verify all kernel pointers are device-accessible before dispatch
                        if (g_ggml_sycl_debug >= 1) {
                            sycl::context sycl_ctx = stream->get_context();
                            auto          q8_alloc =
                                q8_1_buffer ? ggml_sycl_get_alloc_type(q8_1_buffer) : sycl::usm::alloc::unknown;
                            auto dst_alloc = dst_d ? ggml_sycl_get_alloc_type(dst_d) : sycl::usm::alloc::unknown;
                            auto ids_alloc =
                                dispatch_ids ? ggml_sycl_get_alloc_type(dispatch_ids) : sycl::usm::alloc::unknown;
                            auto ptrs_alloc =
                                dispatch_ptrs ? ggml_sycl_get_alloc_type(dispatch_ptrs) : sycl::usm::alloc::unknown;
                            GGML_SYCL_DEBUG(
                                "[MMVQ-USM-CHECK] q8=%d dst=%d ids=%d ptrs=%d dispatch_ids=%p ids_d=%p "
                                "(0=host,1=dev,2=shared,3=unknown)\n",
                                (int) q8_alloc, (int) dst_alloc, (int) ids_alloc, (int) ptrs_alloc,
                                (void *) dispatch_ids, (void *) ids_d);

                            // Check expert pointers inside the table (first few)
                            if (dispatch_ptrs && src0_extra) {
                                const auto & host_ptrs = src0_extra->moe_expert_ptrs_host[ctx.device];
                                for (size_t e = 0; e < std::min(host_ptrs.size(), (size_t) 4); ++e) {
                                    void * eptr   = host_ptrs[e];
                                    auto   ealloc = eptr ? ggml_sycl_get_alloc_type(eptr) : sycl::usm::alloc::unknown;
                                    GGML_SYCL_DEBUG("[MMVQ-USM-CHECK] expert[%zu]=%p alloc=%d\n", e, eptr,
                                                    (int) ealloc);
                                }
                            }
                        }

                        reorder_mul_mat_vec_mxfp4_q8_1_id_sycl(src0_d, dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                                               ne00,     // ncols (MXFP4)
                                                               ne10,     // ncols_y (Q8_1 row size for SoA ds offset)
                                                               ne01,     // nrows_per_expert
                                                               ne02,     // num_experts
                                                               total_batches, n_ids, num_tokens, ne11, ids_nb0,
                                                               ids_nb1,  // ids strides
                                                               q8_nb11, q8_nb12,  // Q8_1 strides
                                                               nb1, nb2,          // dst strides
                                                               stream);
                    }
                } else {
                    // Original AoS layout
                    mul_mat_vec_mxfp4_q8_1_id_sycl(src0_d, dispatch_ptrs, q8_1_buffer, dst_d, dispatch_ids,
                                                   ne00,              // ncols
                                                   ne01,              // nrows_per_expert
                                                   total_batches, n_ids, num_tokens, ne11, stride_expert_x, ids_nb0,
                                                   ids_nb1,           // ids strides
                                                   q8_nb11, q8_nb12,  // Q8_1 strides
                                                   nb1, nb2,          // dst strides
                                                   stream);
                }
            }
            break;
        default:
            GGML_ABORT("Unsupported type for MoE GPU dispatch");
    }

    if (use_ptr_table && src0_extra && ctx.device >= 0 && ctx.device < GGML_SYCL_MAX_DEVICES) {
        ggml_sycl_retain_moe_ptr_table_leases_until_event(const_cast<ggml_tensor_extra_gpu *>(src0_extra), ctx.device,
                                                          stream->ext_oneapi_submit_barrier());
    }

    return true;
}

struct mmvq_stream_segment {
    size_t src_base      = 0;
    size_t bytes_per_row = 0;
};

struct mmvq_stream_ctx {
    int                 device_id            = -1;
    const ggml_tensor * src0                 = nullptr;
    const char *        src1_ddq_i           = nullptr;
    float *             dst_dd_i             = nullptr;
    int64_t             dst_row_stride       = 0;
    int64_t             ne00                 = 0;
    int64_t             ne10                 = 0;
    int64_t             src1_ncols           = 0;
    int64_t             src1_padded_col_size = 0;
    int64_t             row_base             = 0;
    int64_t             total_rows           = 0;
    reorder_mode        mmvq_mode            = reorder_mode::NONE;
    layout_mode         layout               = GGML_LAYOUT_AOS;
    const uint8_t *     src_base             = nullptr;
    size_t              row_total_bytes      = 0;
    int                 segment_count        = 0;
    mmvq_stream_segment segments[4]          = {};
    // Persistent host-pinned staging buffer (owned by ggml_backend_sycl_context).
    // Non-null: reuse across DMA calls without sycl::malloc_host per token.
    void *              host_staging         = nullptr;
    size_t              host_staging_size    = 0;
    // Track the last DMA event that used host_staging so we can wait before
    // overwriting (required when copy_fn is called multiple times per tensor).
    mutable sycl::event prev_staging_evt     = {};
    mutable bool        has_prev_staging_evt = false;
};

static bool mmvq_parse_env_mb_value(const char * name, size_t & out_mb) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    long   mb  = std::strtol(env, &end, 10);
    if (end == env || mb < 0) {
        GGML_LOG_WARN("[MMVQ] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_mb = static_cast<size_t>(mb);
    return true;
}

static bool mmvq_parse_env_count_value(const char * name, size_t & out_count) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return false;
    }
    char * end   = nullptr;
    long   count = std::strtol(env, &end, 10);
    if (end == env || count < 0) {
        GGML_LOG_WARN("[MMVQ] Invalid %s='%s'\n", name, env);
        return false;
    }
    out_count = static_cast<size_t>(count);
    return true;
}

static void mmvq_resolve_dma_params(size_t row_bytes, size_t & slice_bytes, size_t & buffer_count) {
    size_t slice_mb = 1024;
    size_t buffers  = 2;
    size_t env_val  = 0;

    if (mmvq_parse_env_mb_value("GGML_SYCL_DMA_SLICE_MB", env_val)) {
        slice_mb = env_val;
    }
    if (mmvq_parse_env_count_value("GGML_SYCL_DMA_BUFFERS", env_val) ||
        mmvq_parse_env_count_value("GGML_SYCL_DMA_SLICES", env_val)) {
        buffers = env_val;
    }

    if (slice_bytes == 0) {
        slice_bytes = slice_mb * 1024ULL * 1024ULL;
    }
    if (buffer_count == 0) {
        buffer_count = buffers;
    }
    if (row_bytes > 0) {
        size_t rows_per_slice = slice_bytes / row_bytes;
        if (rows_per_slice < 1) {
            rows_per_slice = 1;
        }
        slice_bytes = rows_per_slice * row_bytes;
    }
}

static size_t mmvq_q6_k_coalesced_row_quants_bytes(int blocks_per_row) {
    size_t row_quants_bytes = 0;
    int    remaining        = blocks_per_row;
    while (remaining > 0) {
        int ts = 1;
        while (ts * 2 <= remaining && ts < 32) {
            ts *= 2;
        }
        row_quants_bytes += static_cast<size_t>(ts) * (128 + 64 + 16);
        remaining -= ts;
    }
    return row_quants_bytes;
}

static bool mmvq_build_stream_segments(const ggml_tensor * src0,
                                       layout_mode         layout,
                                       int64_t             ncols,
                                       int64_t             total_rows,
                                       mmvq_stream_ctx &   ctx) {
    ctx.segment_count   = 0;
    ctx.row_total_bytes = 0;
    ctx.total_rows      = total_rows;

    if (layout == GGML_LAYOUT_AOS) {
        ctx.row_total_bytes = ggml_row_size(src0->type, ncols);
        return false;
    }

    const size_t blocks_per_row = static_cast<size_t>(ncols) / ggml_blck_size(src0->type);
    size_t       src_base       = 0;
    auto         add_segment    = [&](size_t bytes_per_row) {
        GGML_ASSERT(ctx.segment_count < 4);
        ctx.segments[ctx.segment_count].src_base      = src_base;
        ctx.segments[ctx.segment_count].bytes_per_row = bytes_per_row;
        ctx.segment_count++;
        src_base += static_cast<size_t>(total_rows) * bytes_per_row;
        ctx.row_total_bytes += bytes_per_row;
    };

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            {
                const size_t q_bytes = static_cast<size_t>(ncols) / 2;
                const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                add_segment(q_bytes);
                add_segment(d_bytes);
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                const size_t q_bytes = static_cast<size_t>(ncols);
                const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                add_segment(q_bytes);
                add_segment(d_bytes);
            }
            break;
        case GGML_TYPE_Q4_K:
            {
                const size_t q_bytes      = blocks_per_row * (QK_K / 2);
                const size_t scales_bytes = blocks_per_row * K_SCALE_SIZE;
                const size_t dm_bytes     = blocks_per_row * 4;
                add_segment(q_bytes);
                add_segment(scales_bytes);
                add_segment(dm_bytes);
            }
            break;
        case GGML_TYPE_Q6_K:
            {
                if (layout == GGML_LAYOUT_COALESCED) {
                    const size_t q_bytes = mmvq_q6_k_coalesced_row_quants_bytes(static_cast<int>(blocks_per_row));
                    const size_t d_bytes = blocks_per_row * sizeof(ggml_half);
                    add_segment(q_bytes);
                    add_segment(d_bytes);
                } else {
                    const size_t ql_bytes     = blocks_per_row * (QK_K / 2);
                    const size_t qh_bytes     = blocks_per_row * (QK_K / 4);
                    const size_t scales_bytes = blocks_per_row * (QK_K / 16);
                    const size_t d_bytes      = blocks_per_row * sizeof(ggml_half);
                    add_segment(ql_bytes);
                    add_segment(qh_bytes);
                    add_segment(scales_bytes);
                    add_segment(d_bytes);
                }
            }
            break;
        case GGML_TYPE_MXFP4:
            {
                const size_t q_bytes = static_cast<size_t>(ncols) / 2;
                const size_t e_bytes = blocks_per_row;
                add_segment(q_bytes);
                add_segment(e_bytes);
            }
            break;
        default:
            GGML_ABORT("MMVQ streaming: unsupported layout/type");
    }

    return true;
}

static void ggml_sycl_mmvq_dispatch(const ggml_tensor *     src0,
                                    const char *            src0_dd_i,
                                    const void *            layout_base,
                                    reorder_mode            mmvq_mode,
                                    int                     device_id,
                                    int64_t                 ne00,
                                    int64_t                 ne01,
                                    int64_t                 ne10,
                                    int64_t                 row_low,
                                    int64_t                 row_high,
                                    int64_t                 src1_ncols,
                                    int64_t                 src1_padded_col_size,
                                    const char *            src1_ddq_i,
                                    float *                 dst_dd_i,
                                    int64_t                 dst_row_stride,
                                    const dpct::queue_ptr & stream) {
    const int64_t row_diff = row_high - row_low;
    const size_t  q8_1_ts  = sizeof(block_q8_1);
    const size_t  q8_1_bs  = QK8_1;

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst_row_stride;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                {
                    reorder_mode mode = mmvq_mode;

                    if (mode == reorder_mode::COALESCED) {
                        GGML_SYCL_KTRACE("mmvq_q4_0_coalesced", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        coalesced_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff,
                                                             stream);
                    } else if (mode == reorder_mode::SOA) {
                        GGML_SYCL_KTRACE("mmvq_q4_0_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        const void * soa_base = layout_base;
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01,
                                                           row_low, stream);
                    } else {
                        GGML_SYCL_KTRACE("mmvq_q4_0_aos", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q4_1:
                GGML_SYCL_KTRACE("mmvq_q4_1", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                GGML_SYCL_KTRACE("mmvq_q5_0", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                GGML_SYCL_KTRACE("mmvq_q5_1", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                {
                    reorder_mode mode = mmvq_mode;

                    // Debug: trace dispatch decision AND verify actual data layout
                    static int q8_0_dispatch_count = 0;
                    if (false && q8_0_dispatch_count < 3) {  // DISABLED for testing
                        const void * data_ptr = layout_base ? layout_base : static_cast<const void *>(src0_dd_i);
                        const size_t nblocks  = (ne00 / QK8_0) * ne01;

                        // Read first 64 bytes from GPU to check layout
                        uint8_t header[64];
                        stream->memcpy(header, data_ptr, sizeof(header)).wait();

                        // AoS expects: [d0:2][qs0:32][d1:2][qs1:32]...
                        // SoA expects: [qs0:32][qs1:32]...[d0:2][d1:2]...

                        // Read d from AoS position (offset 0) and SoA position (offset nblocks*32)
                        ggml_half d_aos = *(ggml_half *) &header[0];

                        // Also check what's at SoA d position (if within buffer)
                        const size_t soa_d_offset   = nblocks * QK8_0;
                        uint8_t      soa_d_bytes[4] = { 0 };
                        if (soa_d_offset < ggml_nbytes(src0)) {
                            stream->memcpy(soa_d_bytes, (const uint8_t *) data_ptr + soa_d_offset, 4).wait();
                        }
                        ggml_half d_soa = *(ggml_half *) &soa_d_bytes[0];

                        fprintf(stderr, "[MMVQ-Q8_0-VALIDATE] %s mode=%d nblocks=%zu\n", src0->name, (int) mode,
                                nblocks);
                        fprintf(stderr, "  Header bytes[0-7]: %02x %02x %02x %02x %02x %02x %02x %02x\n", header[0],
                                header[1], header[2], header[3], header[4], header[5], header[6], header[7]);
                        fprintf(stderr, "  d_aos (offset 0)=%.6f, d_soa (offset %zu)=%.6f\n", (float) d_aos,
                                soa_d_offset, (float) d_soa);
                        fprintf(stderr, "  If mode=SOA, d_soa should be valid (~0.001-1.0), d_aos should be garbage\n");

                        // Also validate Y (Q8_1) data layout
                        // Y SoA layout: [quants: 0..ne00-1][ds: ne00..ne00+nblocks_y*4-1]
                        // Note: ne00 (X cols) should equal ne10 (Y cols) for valid matmul
                        fprintf(stderr, "  Y validation: ne00=%lld (X cols), ne10=%lld (Y cols), src1_padded=%lld\n",
                                (long long) ne00, (long long) ne10, (long long) src1_padded_col_size);

                        // Read first 8 bytes of Y (should be quant values, int8 in range -127 to 127)
                        uint8_t y_header[8] = { 0 };
                        stream->memcpy(y_header, src1_ddq_i_bs, sizeof(y_header)).wait();
                        fprintf(stderr, "  Y quants[0-7]: %d %d %d %d %d %d %d %d\n", (int) (int8_t) y_header[0],
                                (int) (int8_t) y_header[1], (int) (int8_t) y_header[2], (int) (int8_t) y_header[3],
                                (int) (int8_t) y_header[4], (int) (int8_t) y_header[5], (int) (int8_t) y_header[6],
                                (int) (int8_t) y_header[7]);

                        // Read ds at Y SoA position (offset ne00 for first ds)
                        uint8_t y_ds_bytes[4] = { 0 };
                        stream->memcpy(y_ds_bytes, (const uint8_t *) src1_ddq_i_bs + ne00, 4).wait();
                        sycl::half2 y_ds = *(sycl::half2 *) y_ds_bytes;
                        fprintf(stderr, "  Y ds at offset ne00=%lld: d=%.6f, s=%.6f\n", (long long) ne00,
                                (float) y_ds.x(), (float) y_ds.y());

                        // Compare with AoS position (first 4 bytes would be ds if AoS)
                        sycl::half2 y_ds_aos = *(sycl::half2 *) y_header;
                        fprintf(stderr, "  Y ds at offset 0 (AoS): d=%.6f, s=%.6f\n", (float) y_ds_aos.x(),
                                (float) y_ds_aos.y());
                        fprintf(stderr, "  If Y is SoA: ds at ne00 should be valid, ds at 0 should be garbage\n");

                        q8_0_dispatch_count++;
                    }

                    if (mode == reorder_mode::COALESCED) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_q8_0_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q8_0_coalesced", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        coalesced_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff,
                                                             stream);
                    } else if (mode == reorder_mode::SOA) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q8_0_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        // For SoA layout, use base pointer (not pre-offset src0_dd_i) and pass row_low for correct block indexing
                        const void * soa_base = layout_base;
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01,
                                                           row_low, stream);
                    } else {
                        GGML_SYCL_KTRACE("mmvq_q8_0_aos", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_MXFP4:
                {
                    reorder_mode mode = mmvq_mode;

                    if (mode == reorder_mode::COALESCED) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_mxfp4_coalesced", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        coalesced_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff,
                                                              stream);
                    } else if (mode == reorder_mode::SOA) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_mxfp4_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        // For SoA layout, use base pointer (not pre-offset src0_dd_i) and pass row_low for correct block indexing
                        const void * soa_base = layout_base;
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01,
                                                            row_low, stream);
                    } else {
                        GGML_SYCL_KTRACE("mmvq_mxfp4_aos", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q2_K:
                GGML_SYCL_KTRACE("mmvq_q2_k", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                GGML_SYCL_KTRACE("mmvq_q3_k", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                {
                    // Q4_K only has SOA kernel (no COALESCED version) - check SoA specifically
                    if (mmvq_mode == reorder_mode::SOA) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q4_k_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        // For SoA layout, use base pointer (not pre-offset src0_dd_i) and pass row_low for correct block indexing
                        const void * soa_base = layout_base;
                        reorder_mul_mat_vec_q4_k_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01,
                                                           row_low, stream);
                    } else if (mmvq_mode == reorder_mode::COALESCED) {
                        // COALESCED layout not implemented for Q4_K
                        GGML_ABORT("mmvq Q4_K: COALESCED layout not implemented");
                    } else {
                        GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q4_k_aos", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q5_K:
                GGML_SYCL_KTRACE("mmvq_q5_k", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                {
                    // Q6_K supports SOA and COALESCED (variable tile) layouts

#ifdef GGML_SYCL_Q6K_Y_DEBUG
                    // Debug: Verify Y (src1_ddq_i_bs) layout - should be block_q8_1 AoS format
                    // block_q8_1 layout: ds (half2 = 4 bytes) + qs[32] (32 bytes) = 36 bytes per block
                    static int q6k_debug_count = 0;
                    if (q6k_debug_count < 3) {
                        q6k_debug_count++;

                        struct block_q8_1_debug {
                            sycl::half d;
                            sycl::half s;
                            int8_t     qs[32];
                        };

                        // Read first 2 blocks of Y
                        block_q8_1_debug y_blocks[2];
                        stream->memcpy(y_blocks, src1_ddq_i_bs, sizeof(y_blocks)).wait();

                        fprintf(stderr, "\n[Q6K_Y_DEBUG #%d] ne00=%lld row_diff=%lld is_soa=%d\n", q6k_debug_count,
                                (long long) ne00, (long long) row_diff, mmvq_mode == reorder_mode::SOA ? 1 : 0);

                        for (int blk = 0; blk < 2; blk++) {
                            float d_val = static_cast<float>(y_blocks[blk].d);
                            float s_val = static_cast<float>(y_blocks[blk].s);
                            fprintf(stderr, "  Y block[%d]: d=%.6f s=%.6f qs[0..7]=[%d,%d,%d,%d,%d,%d,%d,%d]\n", blk,
                                    d_val, s_val, y_blocks[blk].qs[0], y_blocks[blk].qs[1], y_blocks[blk].qs[2],
                                    y_blocks[blk].qs[3], y_blocks[blk].qs[4], y_blocks[blk].qs[5], y_blocks[blk].qs[6],
                                    y_blocks[blk].qs[7]);
                        }

                        // Also check if Y looks like SoA format (all quants first, then all ds)
                        // If SoA, first 64 bytes would be all qs values (no ds embedded)
                        uint8_t raw_bytes[72];
                        stream->memcpy(raw_bytes, src1_ddq_i_bs, sizeof(raw_bytes)).wait();
                        fprintf(stderr, "  Raw Y[0..35]: ");
                        for (int i = 0; i < 36; i++) {
                            fprintf(stderr, "%02x ", raw_bytes[i]);
                        }
                        fprintf(stderr, "\n  Raw Y[36..71]: ");
                        for (int i = 36; i < 72; i++) {
                            fprintf(stderr, "%02x ", raw_bytes[i]);
                        }
                        fprintf(stderr, "\n");
                    }
#endif

                    if (mmvq_mode == reorder_mode::SOA) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_soa", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        // For SoA layout, use base pointer (not pre-offset src0_dd_i) and pass row_low for correct block indexing
                        const void * soa_base = layout_base;
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(soa_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, ne01,
                                                           row_low, stream);
                    } else if (mmvq_mode == reorder_mode::COALESCED) {
                        GGML_SYCL_DEBUG("Calling coalesced_mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_coalesced", " ne00=%lld row_diff=%lld ne01=%lld row_low=%lld",
                                         (long long) ne00, (long long) row_diff, (long long) ne01, (long long) row_low);
                        // For coalesced layout, use base pointer (not pre-offset src0_dd_i) and pass row_low for correct block indexing
                        const void * coalesced_base = layout_base;
                        coalesced_mul_mat_vec_q6_k_q8_1_sycl(coalesced_base, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff,
                                                             ne01, row_low, stream);
                    } else {
                        GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                        GGML_SYCL_KTRACE("mmvq_q6_k_aos", " ne00=%lld row_diff=%lld", (long long) ne00,
                                         (long long) row_diff);
                        mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_IQ1_S:
                GGML_SYCL_KTRACE("mmvq_iq1_s", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                GGML_SYCL_KTRACE("mmvq_iq1_m", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                GGML_SYCL_KTRACE("mmvq_iq2_xxs", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                GGML_SYCL_KTRACE("mmvq_iq2_xs", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                GGML_SYCL_KTRACE("mmvq_iq2_s", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                GGML_SYCL_KTRACE("mmvq_iq3_xxs", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                GGML_SYCL_KTRACE("mmvq_iq3_s", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                GGML_SYCL_KTRACE("mmvq_iq4_nl", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                GGML_SYCL_KTRACE("mmvq_iq4_xs", " ne00=%lld row_diff=%lld", (long long) ne00, (long long) row_diff);
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }

        // DEBUG: Check output values for attn_output after kernel (layer 0 only)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int mmvq_out_dbg = 0;
        if (g_ggml_sycl_tp_debug && mmvq_out_dbg++ < 10 && src0->name && strstr(src0->name, "blk.0.attn_output")) {
            stream->wait();  // Wait for kernel to complete
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8 * sizeof(float)).wait();
            fprintf(stderr, "TP DEBUG MMVQ output device=%d %s col=%d dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    device_id, src0->name, i, out_vals[0], out_vals[1], out_vals[2], out_vals[3], out_vals[4],
                    out_vals[5], out_vals[6], out_vals[7]);
        }

        // DEBUG: Check FFN down output for layer 0 (works in both TP and single GPU)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int ffn_down_dbg = 0;
        if (g_ggml_sycl_tp_debug && ffn_down_dbg++ < 5 && src0->name && strstr(src0->name, "blk.0.ffn_down")) {
            stream->wait();
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8 * sizeof(float)).wait();
            fprintf(stderr,
                    "DEBUG FFN_DOWN layer0 device=%d col=%d ne00=%lld row_diff=%lld dst[0..7]=[%f, %f, %f, %f, %f, %f, "
                    "%f, %f]\n",
                    device_id, (int) i, (long long) ne00, (long long) row_diff, out_vals[0], out_vals[1], out_vals[2],
                    out_vals[3], out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        }
    }
}

static sycl::event mmvq_stream_copy(sycl::queue &                    queue,
                                    void *                           device_slice,
                                    size_t                           slice_bytes,
                                    size_t                           offset_bytes,
                                    const void *                     src_ptr,
                                    size_t                           src_size,
                                    const void *                     ctx_void,
                                    const std::vector<sycl::event> & deps) {
    GGML_UNUSED(src_ptr);
    GGML_UNUSED(src_size);
    const auto * ctx = static_cast<const mmvq_stream_ctx *>(ctx_void);
    if (!ctx || ctx->segment_count == 0 || ctx->row_total_bytes == 0) {
        return queue.memcpy(device_slice, static_cast<const char *>(src_ptr) + offset_bytes, slice_bytes, deps);
    }
    GGML_ASSERT(offset_bytes % ctx->row_total_bytes == 0);
    GGML_ASSERT(slice_bytes % ctx->row_total_bytes == 0);

    const size_t row_start = offset_bytes / ctx->row_total_bytes;
    const size_t row_count = slice_bytes / ctx->row_total_bytes;
    const size_t src_row   = static_cast<size_t>(ctx->row_base) + row_start;

    const sycl::usm::alloc src_alloc = ggml_sycl_get_alloc_type(ctx->src_base);
    if (src_alloc != sycl::usm::alloc::device) {
        uint8_t * host_slice     = nullptr;
        bool      use_persistent = (ctx->host_staging != nullptr && ctx->host_staging_size >= slice_bytes);
        if (use_persistent) {
            // Wait for the previous DMA that used this staging buffer to complete
            // before overwriting (guards against multi-slice ring-buffer races).
            if (ctx->has_prev_staging_evt) {
                ctx->prev_staging_evt.wait();
                ctx->has_prev_staging_evt = false;
            }
            host_slice = static_cast<uint8_t *>(ctx->host_staging);
        } else {
            host_slice =
                static_cast<uint8_t *>(ggml_sycl_malloc_host_tracked_bytes(slice_bytes, queue, "mmvq:host_stage"));
            if (!host_slice) {
                throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
                                      "MMVQ stream: host staging allocation failed");
            }
        }
        size_t dst_offset = 0;
        for (int i = 0; i < ctx->segment_count; ++i) {
            const auto & seg   = ctx->segments[i];
            const size_t bytes = row_count * seg.bytes_per_row;
            if (bytes == 0) {
                continue;
            }
            const uint8_t * src = ctx->src_base + seg.src_base + src_row * seg.bytes_per_row;
            std::memcpy(host_slice + dst_offset, src, bytes);
            dst_offset += bytes;
        }
        GGML_ASSERT(dst_offset == slice_bytes);
        sycl::event evt = queue.memcpy(device_slice, host_slice, slice_bytes, deps);
        if (use_persistent) {
            ctx->prev_staging_evt     = evt;
            ctx->has_prev_staging_evt = true;
        } else if (auto * cache = ggml_sycl::get_unified_cache(queue)) {
            cache->defer_host_free(host_slice, slice_bytes, evt);
        } else {
            if (!ggml_sycl_graph_recording_active()) {
                evt.wait();
            }
            ggml_sycl_free_host_tracked_bytes(host_slice, slice_bytes, queue);
        }
        return evt;
    }

    size_t                   dst_offset = 0;
    std::vector<sycl::event> cur_deps   = deps;
    sycl::event              last_evt;

    for (int i = 0; i < ctx->segment_count; ++i) {
        const auto & seg   = ctx->segments[i];
        const size_t bytes = row_count * seg.bytes_per_row;
        if (bytes == 0) {
            continue;
        }
        const uint8_t * src = ctx->src_base + seg.src_base + src_row * seg.bytes_per_row;
        void *          dst = static_cast<uint8_t *>(device_slice) + dst_offset;
        last_evt            = queue.memcpy(dst, src, bytes, cur_deps);
        cur_deps.assign(1, last_evt);
        dst_offset += bytes;
    }

    GGML_ASSERT(dst_offset == slice_bytes);
    return last_evt;
}

struct ggml_sycl_mmvq_marker_kernel;

static sycl::event mmvq_stream_slice(sycl::queue &                    queue,
                                     void *                           device_slice,
                                     size_t                           slice_bytes,
                                     size_t                           offset_bytes,
                                     const void *                     ctx_void,
                                     const std::vector<sycl::event> & deps) {
    GGML_UNUSED(deps);
    const auto * ctx = static_cast<const mmvq_stream_ctx *>(ctx_void);
    if (!ctx || ctx->row_total_bytes == 0) {
        return ggml_sycl_submit_marker<ggml_sycl_mmvq_marker_kernel>(queue);
    }
    GGML_ASSERT(offset_bytes % ctx->row_total_bytes == 0);
    GGML_ASSERT(slice_bytes % ctx->row_total_bytes == 0);

    const size_t row_start = offset_bytes / ctx->row_total_bytes;
    const size_t row_count = slice_bytes / ctx->row_total_bytes;
    float *      dst_ptr   = ctx->dst_dd_i + row_start;

    const void * layout_ptr =
        (ctx->mmvq_mode == reorder_mode::SOA || ctx->mmvq_mode == reorder_mode::COALESCED) ? device_slice : nullptr;

    ggml_sycl_mmvq_dispatch(ctx->src0, static_cast<const char *>(device_slice), layout_ptr, ctx->mmvq_mode,
                            ctx->device_id, ctx->ne00, static_cast<int64_t>(row_count), ctx->ne10, 0,
                            static_cast<int64_t>(row_count), ctx->src1_ncols, ctx->src1_padded_col_size,
                            ctx->src1_ddq_i, dst_ptr, ctx->dst_row_stride, &queue);

    return ggml_sycl_submit_marker<ggml_sycl_mmvq_marker_kernel>(queue);
}

// =============================================================================
// Direct MMVQ kernel submission wrappers for persistent TG micro-graph
// =============================================================================
// These submit standalone MMVQ kernels without ggml_tensor metadata.
// Weights must be in SOA layout.  Activations in SOA Q8_1 format.

// Kernel name tags (distinct from the static-function kernel names above)
class mmvq_persistent_q4_0_tag;
class mmvq_persistent_q6_k_tag;
class mmvq_persistent_quantize_tag;

void mmvq_submit_q4_0_soa(sycl::queue & q,
                          const void *  weights_soa,
                          const void *  y_q8_soa,
                          float *       dst,
                          int           ncols,
                          int           nrows,
                          int           total_nrows,
                          int           row_low) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_persistent_q4_0_tag>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(weights_soa, y_q8_soa, dst, ncols, nrows,
                                                                              total_nrows, row_low, nd_item);
            });
    });
}

void mmvq_submit_q6_k_soa(sycl::queue & q,
                          const void *  weights_soa,
                          const void *  y_q8_soa,
                          float *       dst,
                          int           ncols,
                          int           nrows,
                          int           total_nrows,
                          int           row_low) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_persistent_q6_k_tag>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(weights_soa, y_q8_soa, dst, ncols, nrows,
                                                                              total_nrows, row_low, nd_item);
            });
    });
}

class mmvq_persistent_mxfp4_tag;

void mmvq_submit_mxfp4_soa(sycl::queue & q,
                           const void *  weights_soa,
                           const void *  y_q8_soa,
                           float *       dst,
                           int           ncols,
                           int           nrows,
                           int           total_nrows,
                           int           row_low) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_persistent_mxfp4_tag>(
            sycl::nd_range<3>(global_size, workgroup_size),
            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(weights_soa, y_q8_soa, dst, ncols, nrows,
                                                                               total_nrows, row_low, nd_item);
            });
    });
}

void mmvq_submit_quantize_q8_1_soa(sycl::queue & q, const float * x, void * y_q8_soa, int ncols) {
    // Single-row quantization (M=1 for TG).
    // Uses quantize_and_reorder_q8_1_soa to produce SOA layout:
    //   quants at [0, ncols), ds at [ncols, ncols + ncols/QK8_1 * 4)
    const int  ky           = 1;
    const int  kx_padded    = ncols;  // No padding for M=1
    const int  num_blocks   = ky * (ncols / QK8_1);
    const auto local_range  = std::size_t(WARP_SIZE);
    const auto global_range = num_blocks * local_range;

    q.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_persistent_quantize_tag>(
            sycl::nd_range<1>({ global_range }, { local_range }),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                quantize_and_reorder_q8_1_soa<QK8_1 / WARP_SIZE>()(x, y_q8_soa, ncols, kx_padded, it);
            });
    });
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx,
                                const ggml_tensor *         src0,
                                const ggml_tensor *         src1,
                                ggml_tensor *               dst,
                                const char *                src0_dd_i,
                                const float *               src1_ddf_i,
                                const char *                src1_ddq_i,
                                float *                     dst_dd_i,
                                const int64_t               row_low,
                                const int64_t               row_high,
                                const int64_t               src1_ncols,
                                const int64_t               src1_padded_col_size,
                                const dpct::queue_ptr &     stream) {
    GGML_SYCL_PROFILE_SCOPE_MMVQ("mmvq");

    static bool mmvq_inner_timing_checked = false;
    static bool mmvq_inner_timing         = false;
    if (!mmvq_inner_timing_checked) {
        mmvq_inner_timing         = std::getenv("GGML_SYCL_MMVQ_INNER_TIMING") != nullptr;
        mmvq_inner_timing_checked = true;
    }
    std::chrono::steady_clock::time_point mmvq_inner_t0;
    if (mmvq_inner_timing) {
        mmvq_inner_t0 = std::chrono::steady_clock::now();
    }

    // Check tiered dispatch for weight tensor (debug only)
    if (g_ggml_sycl_debug && src0->name && g_tiered_enabled.load(std::memory_order_relaxed)) {
        ggml_sycl::memory_tier tier;
        bool                   in_inventory = false;
        void *                 cached_ptr   = get_cached_tensor_ptr(src0->name, &tier, &in_inventory);
        if (cached_ptr != nullptr) {
            GGML_SYCL_DEBUG("[SYCL] mmvq tiered hit: %s (tier=%d)\n", src0->name, static_cast<int>(tier));
        } else if (in_inventory) {
            GGML_SYCL_DEBUG("[SYCL] mmvq tiered pending: %s\n", src0->name);
        }
    }

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t ne01     = src0->ne[1];  // Total tensor rows, needed for SoA offset calculations
    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));

    reorder_mode mmvq_mode   = reorder_mode::NONE;
    const void * layout_base = nullptr;
    layout_mode  layout      = GGML_LAYOUT_AOS;

    // Unified weight resolution: single O(1) cache lookup
    auto resolved = ggml_sycl_resolve(src0, device_id);
    if (resolved && (resolved.layout == GGML_LAYOUT_SOA || resolved.layout == GGML_LAYOUT_COALESCED)) {
        layout      = resolved.layout;
        layout_base = resolved.ptr;
        mmvq_mode   = (layout == GGML_LAYOUT_SOA) ? reorder_mode::SOA : reorder_mode::COALESCED;
    }

    if (mmvq_mode != reorder_mode::NONE && !layout_base) {
        GGML_SYCL_DEBUG("[MMVQ] Missing layout pointer for %s layout=%d, falling back to AoS\n",
                        src0->name ? src0->name : "?", (int) layout);
        mmvq_mode = reorder_mode::NONE;
    }
    std::chrono::steady_clock::time_point mmvq_inner_resolve_t1;
    if (mmvq_inner_timing) {
        mmvq_inner_resolve_t1 = std::chrono::steady_clock::now();
    }
    const layout_mode dispatch_layout = (mmvq_mode == reorder_mode::NONE) ? GGML_LAYOUT_AOS : layout;
    const void *      dispatch_base =
        (mmvq_mode == reorder_mode::SOA || mmvq_mode == reorder_mode::COALESCED) ? layout_base : nullptr;
    const char *  dispatch_ptr   = src0_dd_i;
    const int64_t dst_row_stride = dst->ne[0];

    mmvq_stream_ctx stream_ctx{};
    const bool      custom_copy     = mmvq_build_stream_segments(src0, dispatch_layout, ne00, ne01, stream_ctx);
    stream_ctx.device_id            = device_id;
    stream_ctx.src0                 = src0;
    stream_ctx.src1_ddq_i           = src1_ddq_i;
    stream_ctx.dst_dd_i             = dst_dd_i;
    stream_ctx.dst_row_stride       = dst_row_stride;
    stream_ctx.ne00                 = ne00;
    stream_ctx.ne10                 = ne10;
    stream_ctx.src1_ncols           = src1_ncols;
    stream_ctx.src1_padded_col_size = src1_padded_col_size;
    stream_ctx.row_base             = row_low;
    stream_ctx.mmvq_mode            = mmvq_mode;
    stream_ctx.layout               = dispatch_layout;

    auto infer_location = [&](const void * ptr) -> ggml_sycl::cache_location {
        if (!ptr) {
            return ggml_sycl::cache_location::HOST_MMAP;
        }
        const sycl::usm::alloc alloc = ggml_sycl_get_alloc_type(ptr);
        if (alloc == sycl::usm::alloc::device) {
            return ggml_sycl::cache_location::DEVICE;
        }
        if (alloc == sycl::usm::alloc::host || alloc == sycl::usm::alloc::shared) {
            return ggml_sycl::cache_location::HOST_PINNED;
        }
        return ggml_sycl::cache_location::HOST_MMAP;
    };

    ggml_sycl::unified_cache * cache =
        ggml_sycl::unified_cache_enabled() ? ggml_sycl::get_unified_cache(*stream) : nullptr;
    ggml_sycl_cache_id cache_key =
        cache ? ggml_backend_sycl_get_weight_cache_key(src0, device_id) : ggml_sycl_cache_id{};
    ggml_sycl::cache_ptr_view view{};
    if (cache && cache_key.valid) {
        view = cache->get_view(cache_key, dispatch_layout);
    }
    const void * view_ptr = dispatch_base ? dispatch_base : dispatch_ptr;
    if (!view.ptr) {
        view.ptr      = const_cast<void *>(view_ptr);
        view.size     = stream_ctx.row_total_bytes * static_cast<size_t>(ne01);
        view.layout   = dispatch_layout;
        view.type     = ggml_sycl::cache_entry_type::DENSE_WEIGHT;
        view.location = infer_location(view.ptr);
    }
    stream_ctx.src_base = static_cast<const uint8_t *>(view.ptr);

    if (view.ptr && view.location != ggml_sycl::cache_location::DEVICE) {
        if (!cache) {
            GGML_ABORT("MMVQ streaming requires unified cache");
        }
        size_t slice_bytes  = 0;
        size_t buffer_count = 0;
        mmvq_resolve_dma_params(stream_ctx.row_total_bytes, slice_bytes, buffer_count);
        const size_t total_bytes = stream_ctx.row_total_bytes * static_cast<size_t>(row_diff);
        // Pre-wire persistent host staging so copy_fn avoids per-call sycl::malloc_host.
        // Grow only if the pre-resolved slice is larger than the current buffer.
        if (custom_copy && slice_bytes > 0) {
            void * stg                   = ctx.ensure_mmvq_host_staging(slice_bytes, *stream);
            stream_ctx.host_staging      = stg;
            stream_ctx.host_staging_size = stg ? slice_bytes : 0;
        }
        if (cache_key.valid) {
            cache->pin(cache_key, dispatch_layout);
        }
        auto result = cache->stream_dma(view, total_bytes, slice_bytes, buffer_count, mmvq_stream_slice, &stream_ctx,
                                        {}, custom_copy ? mmvq_stream_copy : nullptr);
        if (cache_key.valid) {
            if (result.ok) {
                cache->unpin_on_event(cache_key, dispatch_layout, result.event);
            } else {
                cache->unpin(cache_key, dispatch_layout);
            }
        }
        if (!result.ok) {
            if (result.mmap_direct_failed) {
                GGML_LOG_WARN("[MMVQ] DMA from mmap failed, falling back to CPU (%s)\n",
                              src0->name ? src0->name : "unknown");
                if (ggml_sycl_cpu_fallback_graph(ctx, dst, "mmvq streaming")) {
                    return;
                }
            }
            GGML_ABORT("MMVQ streaming failed");
        }
        GGML_UNUSED(src1);
        GGML_UNUSED(src1_ddf_i);
        GGML_UNUSED(ctx);
        return;
    }

    std::chrono::steady_clock::time_point mmvq_inner_dispatch_t0;
    if (mmvq_inner_timing) {
        mmvq_inner_dispatch_t0 = std::chrono::steady_clock::now();
    }
    ggml_sycl_mmvq_dispatch(src0, dispatch_ptr, dispatch_base, mmvq_mode, device_id, ne00, ne01, ne10, row_low,
                            row_high, src1_ncols, src1_padded_col_size, src1_ddq_i, dst_dd_i, dst_row_stride, stream);
    if (mmvq_inner_timing) {
        stream->wait();
        const auto   mmvq_inner_t1 = std::chrono::steady_clock::now();
        const double resolve_ms =
            std::chrono::duration<double, std::milli>(mmvq_inner_resolve_t1 - mmvq_inner_t0).count();
        const double setup_ms =
            std::chrono::duration<double, std::milli>(mmvq_inner_dispatch_t0 - mmvq_inner_resolve_t1).count();
        const double dispatch_wait_ms =
            std::chrono::duration<double, std::milli>(mmvq_inner_t1 - mmvq_inner_dispatch_t0).count();
        fprintf(stderr,
                "[MMVQ-INNER] resolve=%6.3f setup=%6.3f dispatch_wait=%6.3f src0=%s type=%d layout=%d mode=%d "
                "rows=[%lld,%lld) ne00=%lld ne01=%lld src1_cols=%lld padded=%lld\n",
                resolve_ms, setup_ms, dispatch_wait_ms, src0->name ? src0->name : "?", (int) src0->type,
                (int) dispatch_layout, (int) mmvq_mode, (long long) row_low, (long long) row_high, (long long) ne00,
                (long long) ne01, (long long) src1_ncols, (long long) src1_padded_col_size);
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

namespace ggml_sycl {

bool ggml_sycl_mmvq_bench_launch(const mmvq_bench_args & args, std::vector<sycl::event> * events) {
    if (!args.stream || !args.weights || !args.activations || !args.output) {
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        return false;
    }
    if (args.ncols % QK8_1 != 0) {
        return false;
    }
    if (args.row_low < 0 || args.row_high <= args.row_low || args.row_high > args.nrows) {
        return false;
    }
    if (args.src1_padded_col_size <= 0 || args.dst_row_stride <= 0) {
        return false;
    }

    reorder_mode mmvq_mode   = reorder_mode::NONE;
    const void * layout_base = nullptr;
    switch (args.layout) {
        case GGML_LAYOUT_AOS:
            mmvq_mode = reorder_mode::NONE;
            break;
        case GGML_LAYOUT_SOA:
            mmvq_mode   = reorder_mode::SOA;
            layout_base = args.layout_base;
            if (!layout_base) {
                return false;
            }
            break;
        case GGML_LAYOUT_COALESCED:
            mmvq_mode   = reorder_mode::COALESCED;
            layout_base = args.layout_base;
            if (!layout_base) {
                return false;
            }
            break;
        default:
            return false;
    }

    int device_id = args.device_id;
    if (device_id < 0) {
        SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));
    }

    ggml_tensor fake{};
    fake.type    = args.weight_type;
    fake.name[0] = '\0';

    // MMVQ kernels do not currently surface per-kernel events; ignore events.
    (void) events;

    ggml_sycl_mmvq_dispatch(&fake, static_cast<const char *>(args.weights), layout_base, mmvq_mode, device_id,
                            args.ncols, args.nrows, args.ncols, args.row_low, args.row_high, args.batch,
                            args.src1_padded_col_size, static_cast<const char *>(args.activations), args.output,
                            args.dst_row_stride, args.stream);
    return true;
}

bool ggml_sycl_mxfp4_pair_glu_bench_launch(const mxfp4_pair_glu_bench_args & args) {
    if (!args.stream || !args.gate_ptrs || !args.up_ptrs || !args.activations_q8_soa || !args.output || !args.ids) {
        return false;
    }
    if (args.ncols <= 0 || args.ncols_y <= 0 || args.nrows_per_expert <= 0 || args.n_ids <= 0 || args.n_tokens <= 0 ||
        args.ne11 <= 0) {
        return false;
    }
    if (args.ncols % QK_MXFP4 != 0 || args.ids_nb0 <= 0 || args.ids_nb1 <= 0 || args.nb11 <= 0 || args.nb12 <= 0 ||
        args.dst_nb1 <= 0 || args.dst_nb2 <= 0) {
        return false;
    }
    if (args.glu_op != GGML_GLU_OP_SWIGLU && args.glu_op != GGML_GLU_OP_SWIGLU_OAI) {
        return false;
    }

    const int total_batches = args.n_ids * args.n_tokens;
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (args.direct_xmx) {
        reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_xmx_sycl(
            args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias, args.up_bias,
            args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens, args.ne11,
            args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.gate_bias_nb1,
            args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.stream);
        return true;
    }
#else
    if (args.direct_xmx) {
        return false;
    }
#endif
    switch (args.rows_per_wg) {
        case 1:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<1, false>(
                    args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                    args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids,
                    args.n_tokens, args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1,
                    args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit,
                    args.cache_y, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<1, true>(
                args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens,
                args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2,
                args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.cache_y, args.stream);
            return true;
        case 2:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<2, false>(
                    args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                    args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids,
                    args.n_tokens, args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1,
                    args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit,
                    args.cache_y, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<2, true>(
                args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens,
                args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2,
                args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.cache_y, args.stream);
            return true;
        case 4:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<4, false>(
                    args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                    args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids,
                    args.n_tokens, args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1,
                    args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit,
                    args.cache_y, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<4, true>(
                args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens,
                args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2,
                args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.cache_y, args.stream);
            return true;
        case 8:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<8, false>(
                    args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                    args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids,
                    args.n_tokens, args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1,
                    args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit,
                    args.cache_y, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<8, true>(
                args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens,
                args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2,
                args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.cache_y, args.stream);
            return true;
        case 16:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<16, false>(
                    args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                    args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids,
                    args.n_tokens, args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1,
                    args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit,
                    args.cache_y, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_pair_glu_sycl_rows<16, true>(
                args.gate_ptrs, args.up_ptrs, args.activations_q8_soa, args.output, args.ids, args.gate_bias,
                args.up_bias, args.ncols, args.ncols_y, args.nrows_per_expert, total_batches, args.n_ids, args.n_tokens,
                args.ne11, args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2,
                args.gate_bias_nb1, args.up_bias_nb1, args.glu_op, args.alpha, args.limit, args.cache_y, args.stream);
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_mxfp4_mmv_id_bench_launch(const mxfp4_mmv_id_bench_args & args) {
    if (!args.stream || !args.expert_ptrs || !args.activations_q8_soa || !args.output || !args.ids) {
        return false;
    }
    if (args.ncols <= 0 || args.ncols_y <= 0 || args.nrows_per_expert <= 0 || args.num_experts <= 0 ||
        args.n_ids <= 0 || args.n_tokens <= 0 || args.ne11 <= 0) {
        return false;
    }
    if (args.ncols % QK_MXFP4 != 0 || args.ids_nb0 <= 0 || args.ids_nb1 <= 0 || args.nb11 <= 0 || args.nb12 <= 0 ||
        args.dst_nb1 <= 0 || args.dst_nb2 <= 0) {
        return false;
    }

    const int total_batches = args.n_ids * args.n_tokens;
    switch (args.rows_per_wg) {
        case 1:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<1, false>(
                    nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                    args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                    args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<1, true>(
                nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
            return true;
        case 2:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<2, false>(
                    nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                    args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                    args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<2, true>(
                nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
            return true;
        case 4:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<4, false>(
                    nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                    args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                    args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<4, true>(
                nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
            return true;
        case 8:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<8, false>(
                    nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                    args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                    args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<8, true>(
                nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
            return true;
        case 16:
            if (args.ignore_weight_scale) {
                reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<16, false>(
                    nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                    args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                    args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
                return true;
            }
            reorder_mul_mat_vec_mxfp4_q8_1_id_sycl_rows<16, true>(
                nullptr, args.expert_ptrs, args.activations_q8_soa, args.output, args.ids, args.ncols, args.ncols_y,
                args.nrows_per_expert, args.num_experts, total_batches, args.n_ids, args.n_tokens, args.ne11,
                args.ids_nb0, args.ids_nb1, args.nb11, args.nb12, args.dst_nb1, args.dst_nb2, args.stream);
            return true;
        default:
            return false;
    }
}

}  // namespace ggml_sycl
