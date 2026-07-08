#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#if __has_include(<sycl/ext/intel/esimd.hpp>) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    include <sycl/ext/intel/esimd.hpp>
#    include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#else
#    error "SYCL MXFP4 source-line probe requires Intel ESIMD DPAS headers"
#endif

#ifdef SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY

static constexpr int GGML_SYCL_MXFP4_MOE_XMX_N_SOURCE_LINE = 16;
static constexpr int GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE = 32;

class mxfp4_dpas_pack_q8_source_line_kernel;
#    line 9107 "ggml/src/ggml-sycl/mmvq.cpp"
template <int Repeat, int GLU_OP, bool Prefetch, bool TG1Index> struct mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel;

#    line 6754 "ggml/src/ggml-sycl/mmvq.cpp"

static sycl::event mxfp4_dpas_pack_q8_source_line_sycl(sycl::queue & queue,
                                                       const void *  q8_src,
                                                       int8_t *      b_packed,
                                                       float *       y_scales,
                                                       int           ncols,
                                                       int           ncols_y,
                                                       int           groups,
                                                       int           n_tokens,
                                                       int           ne11,
                                                       int64_t       q8_nb11,
                                                       int64_t       q8_nb12) {
    constexpr int exec_n  = GGML_SYCL_MXFP4_MOE_XMX_N_SOURCE_LINE;
    constexpr int k_per   = GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE;
    const int     k_tiles = ncols / k_per;

    return queue.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mxfp4_dpas_pack_q8_source_line_kernel>(
            sycl::range<1>(static_cast<size_t>(groups) * static_cast<size_t>(k_tiles) * static_cast<size_t>(k_per)),
            [=](sycl::id<1> idx) {
                const int64_t linear = static_cast<int64_t>(idx[0]);
                const int     kk     = static_cast<int>(linear % k_per);
                const int64_t t0     = linear / k_per;
                const int     kt     = static_cast<int>(t0 % k_tiles);
                const int     group  = static_cast<int>(t0 / k_tiles);
                const int     id     = group / n_tokens;
                const int     iid1   = group - id * n_tokens;
                const int64_t i11    = id % ne11;
                const int64_t i12    = iid1;

                const char * q8_row = static_cast<const char *>(q8_src) + i11 * q8_nb11 + i12 * q8_nb12;
                const int8_t q      = reinterpret_cast<const int8_t *>(q8_row)[kt * k_per + kk];

                const size_t tile_base =
                    (static_cast<size_t>(group) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) *
                    static_cast<size_t>(k_per * exec_n);
                const int vnni_offset                                  = (kk / 4) * exec_n * 4 + (kk % 4);
                b_packed[tile_base + static_cast<size_t>(vnni_offset)] = q;

                if (kk == 0) {
                    const auto * ds =
                        reinterpret_cast<const sycl::half2 *>(q8_row + ncols_y + kt * sizeof(sycl::half2));
                    y_scales[(static_cast<size_t>(group) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) *
                             static_cast<size_t>(exec_n)] = static_cast<float>(ds->x());
                }
            });
    });
}

#    line 7077 "ggml/src/ggml-sycl/mmvq.cpp"

template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<float, N> mxfp4_e8m0_to_fp32_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> e) {
    using namespace sycl::ext::intel::esimd;
    simd<uint32_t, N> bits = (convert<uint32_t>(e) - uint32_t{ 1 }) << 23;
    bits.merge(simd<uint32_t, N>(0x00200000u), e == uint8_t{ 0 });
    bits.merge(simd<uint32_t, N>(0x00400000u), e == uint8_t{ 1 });
    return bits.template bit_cast_view<float>();
}

#    line 7087 "ggml/src/ggml-sycl/mmvq.cpp"

template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<int8_t, N> mxfp4_code_values_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> codes) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, N> base_mag = codes & uint8_t{ 7 };
    simd<uint8_t, N> extra    = base_mag - uint8_t{ 4 };
    extra.merge(simd<uint8_t, N>(0), base_mag <= uint8_t{ 4 });
    simd<uint8_t, N> mag = base_mag + extra;
    mag.merge(simd<uint8_t, N>(12), base_mag == uint8_t{ 7 });

    simd<int8_t, N> values = mag;
    simd<int8_t, N> neg    = -values;
    values.merge(neg, (codes & uint8_t{ 8 }) != uint8_t{ 0 });
    return values;
}

#    line 7103 "ggml/src/ggml-sycl/mmvq.cpp"

template <int GLU_OP>
SYCL_ESIMD_FUNCTION inline float mmvq_moe_apply_pair_glu_esimd(float       gate,
                                                               float       up,
                                                               const float alpha,
                                                               const float limit) {
    using namespace sycl::ext::intel::esimd;
    if constexpr (GLU_OP == GGML_GLU_OP_SWIGLU_OAI) {
        const float gate_limited = gate < limit ? gate : limit;
        float       up_limited   = up < limit ? up : limit;
        up_limited               = up_limited > -limit ? up_limited : -limit;
        return (gate_limited / (1.0f + exp(-gate_limited * alpha))) * (1.0f + up_limited);
    }
    return (gate / (1.0f + exp(-gate))) * up;
}

#    line 7219 "ggml/src/ggml-sycl/mmvq.cpp"

template <int Repeat>
SYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec_from_group(
    const uint8_t *                                                                         group,
    int64_t                                                                                 tile_n_total,
    int64_t                                                                                 xmx_row_in_group,
    sycl::ext::intel::esimd::simd<int8_t, Repeat * GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE> & a_vec,
    sycl::ext::intel::esimd::simd<float, Repeat> &                                          w_scale_vec) {
    using namespace sycl::ext::intel::esimd;
    constexpr int k_per         = GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE;
    constexpr int packed_bytes  = k_per / 2;
    constexpr int compact_bytes = Repeat * packed_bytes;

    const uint8_t * scale_ptr  = group + xmx_row_in_group;
    const uint8_t * packed_ptr = group + tile_n_total + xmx_row_in_group * packed_bytes;

    simd<uint8_t, Repeat>        scale_bytes = block_load<uint8_t, Repeat>(scale_ptr);
    simd<uint8_t, compact_bytes> packed      = block_load<uint8_t, compact_bytes>(packed_ptr);
    w_scale_vec                              = mxfp4_e8m0_to_fp32_esimd<Repeat>(scale_bytes);
#    pragma unroll
    for (int r = 0; r < Repeat; ++r) {
        simd<uint8_t, packed_bytes> row = packed.template select<packed_bytes, 1>(r * packed_bytes);
        simd<uint8_t, k_per>        codes;
        codes.template select<packed_bytes, 1>(0)            = row & uint8_t{ 0x0f };
        codes.template select<packed_bytes, 1>(packed_bytes) = row >> 4;
        a_vec.template select<k_per, 1>(r * k_per)           = mxfp4_code_values_esimd<k_per>(codes);
    }
}

#    line 9730 "ggml/src/ggml-sycl/mmvq.cpp"

template <int Repeat, int GLU_OP, bool Prefetch, bool TG1Index>
static sycl::event mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl(sycl::queue &        queue,
                                                         const void * const * gate_ptrs,
                                                         const void * const * up_ptrs,
                                                         const int8_t *       b_packed,
                                                         const float *        y_scales,
                                                         float *              dst_glu,
                                                         const int32_t *      ids,
                                                         const float *        gate_bias,
                                                         const float *        up_bias,
                                                         int                  ncols,
                                                         int                  nrows_per_expert,
                                                         int                  total_batches,
                                                         int                  n_tokens,
                                                         int64_t              ids_nb0,
                                                         int64_t              ids_nb1,
                                                         int64_t              dst_nb1,
                                                         int64_t              dst_nb2,
                                                         int64_t              gate_bias_nb1,
                                                         int64_t              up_bias_nb1,
                                                         float                alpha,
                                                         float                limit,
                                                         int                  tile_n_total,
                                                         const sycl::event &  pack_event) {
    static_assert(!Prefetch, "source-line MXFP4 m2 probe does not compile the prefetch variant");
    static_assert(!TG1Index, "source-line MXFP4 m2 probe does not compile the TG1 index variant");
    constexpr int exec_n = GGML_SYCL_MXFP4_MOE_XMX_N_SOURCE_LINE;
    constexpr int k_per  = GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE;
    constexpr int an     = Repeat * k_per;
    constexpr int bn     = k_per * exec_n;

    const int64_t m_tiles      = (static_cast<int64_t>(nrows_per_expert) + Repeat - 1) / Repeat;
    const int64_t m_tile_pairs = (m_tiles + 1) / 2;
    const int64_t k_tiles      = ncols / k_per;
    const int64_t tiles        = static_cast<int64_t>(total_batches) * m_tile_pairs;

    // clang-format off
    return queue.submit([&](sycl::handler & h) {
        h.depends_on(pack_event);
        h.parallel_for<mxfp4_pair_glu_xmx_tiled_dpas_m2_kernel<Repeat, GLU_OP, Prefetch, TG1Index>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tiles)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                using namespace sycl::ext::intel::esimd;
                const int64_t tile_idx = static_cast<int64_t>(item.get_global_id(0));
                const int64_t group    = tile_idx / m_tile_pairs;
                const int64_t pair_m   = tile_idx - group * m_tile_pairs;
                const int64_t tile_m0  = pair_m * 2;
                const int64_t tile_m1  = tile_m0 + 1;
                const bool    have_m1  = tile_m1 < m_tiles;
                const int     id       = static_cast<int>(group / n_tokens);
                const int     iid1     = static_cast<int>(group - static_cast<int64_t>(id) * n_tokens);
                const int32_t expert_id =
                    ids ? *(const int32_t *) ((const char *) ids + static_cast<int64_t>(iid1) * ids_nb1 +
                                              static_cast<int64_t>(id) * ids_nb0) :
                          static_cast<int32_t>(group);

                const uint8_t * gate_base = reinterpret_cast<const uint8_t *>(gate_ptrs[expert_id]);
                const uint8_t * up_base   = reinterpret_cast<const uint8_t *>(up_ptrs[expert_id]);
                if (!gate_base || !up_base) {
                    return;
                }

                const int64_t n_tile_groups_n = (nrows_per_expert + tile_n_total - 1) / tile_n_total;
                const int64_t group_bytes     = tile_n_total * (1 + k_per / 2);
                const int64_t kt_group_stride = n_tile_groups_n * group_bytes;

                const int64_t xmx_row_start0    = tile_m0 * Repeat;
                const int64_t xmx_group_n0      = xmx_row_start0 / tile_n_total;
                const int64_t xmx_row_in_group0 = xmx_row_start0 - xmx_group_n0 * tile_n_total;
                const int64_t xmx_row_start1    = tile_m1 * Repeat;
                const int64_t xmx_group_n1      = xmx_row_start1 / tile_n_total;
                const int64_t xmx_row_in_group1 = xmx_row_start1 - xmx_group_n1 * tile_n_total;

                const uint8_t * gate_group0 = gate_base + xmx_group_n0 * group_bytes;
                const uint8_t * up_group0   = up_base + xmx_group_n0 * group_bytes;
                const uint8_t * gate_group1 = gate_base + xmx_group_n1 * group_bytes;
                const uint8_t * up_group1   = up_base + xmx_group_n1 * group_bytes;
                const int8_t *  b_ptr       = b_packed + (group * k_tiles) * bn;

                simd<float, Repeat> gate_acc0 = 0.0f;
                simd<float, Repeat> up_acc0   = 0.0f;
                simd<float, Repeat> gate_acc1 = 0.0f;
                simd<float, Repeat> up_acc1   = 0.0f;
                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    simd<int8_t, bn> b_vec   = block_load<int8_t, bn>(b_ptr);
                    simd<float, 1>   y_scale = block_load<float, 1>(y_scales + (group * k_tiles + kt) * exec_n);

                    simd<int8_t, an>    gate_a_vec0;
                    simd<int8_t, an>    up_a_vec0;
                    simd<float, Repeat> gate_w_scale0;
                    simd<float, Repeat> up_w_scale0;
                    mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group0, tile_n_total, xmx_row_in_group0,
                                                                  gate_a_vec0, gate_w_scale0);
                    mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group0, tile_n_total, xmx_row_in_group0, up_a_vec0,
                                                                  up_w_scale0);

                    simd<int, Repeat * exec_n> gate_part0 = 0;
                    simd<int, Repeat * exec_n> up_part0   = 0;
                    gate_part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(gate_part0, b_vec, gate_a_vec0);
                    up_part0   = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(up_part0, b_vec, up_a_vec0);
#pragma unroll
                    for (int r = 0; r < Repeat; ++r) {
                        simd<int, 1>   gate_i = gate_part0.template select<1, 1>(r * exec_n);
                        simd<int, 1>   up_i   = up_part0.template select<1, 1>(r * exec_n);
                        simd<float, 1> gate_f = convert<float>(gate_i) * (y_scale * gate_w_scale0[r]);
                        simd<float, 1> up_f   = convert<float>(up_i) * (y_scale * up_w_scale0[r]);
                        gate_acc0[r] += gate_f[0];
                        up_acc0[r] += up_f[0];
                    }

                    if (have_m1) {
                        simd<int8_t, an>    gate_a_vec1;
                        simd<int8_t, an>    up_a_vec1;
                        simd<float, Repeat> gate_w_scale1;
                        simd<float, Repeat> up_w_scale1;
                        mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(gate_group1, tile_n_total, xmx_row_in_group1,
                                                                      gate_a_vec1, gate_w_scale1);
                        mxfp4_xmx_tiled_load_a_vec_from_group<Repeat>(up_group1, tile_n_total, xmx_row_in_group1,
                                                                      up_a_vec1, up_w_scale1);

                        simd<int, Repeat * exec_n> gate_part1 = 0;
                        simd<int, Repeat * exec_n> up_part1   = 0;
                        gate_part1 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(gate_part1, b_vec, gate_a_vec1);
                        up_part1   = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(up_part1, b_vec, up_a_vec1);
#pragma unroll
                        for (int r = 0; r < Repeat; ++r) {
                            simd<int, 1>   gate_i = gate_part1.template select<1, 1>(r * exec_n);
                            simd<int, 1>   up_i   = up_part1.template select<1, 1>(r * exec_n);
                            simd<float, 1> gate_f = convert<float>(gate_i) * (y_scale * gate_w_scale1[r]);
                            simd<float, 1> up_f   = convert<float>(up_i) * (y_scale * up_w_scale1[r]);
                            gate_acc1[r] += gate_f[0];
                            up_acc1[r] += up_f[0];
                        }
                    }

                    b_ptr += bn;
                    gate_group0 += kt_group_stride;
                    up_group0 += kt_group_stride;
                    gate_group1 += kt_group_stride;
                    up_group1 += kt_group_stride;
                }

                float * dst_out =
                    reinterpret_cast<float *>(reinterpret_cast<char *>(dst_glu) + static_cast<int64_t>(id) * dst_nb1 +
                                              static_cast<int64_t>(iid1) * dst_nb2);
#pragma unroll
                for (int r = 0; r < Repeat; ++r) {
                    const int row = static_cast<int>(tile_m0) * Repeat + r;
                    if (row < nrows_per_expert) {
                        float gate_value = gate_acc0[r];
                        float up_value   = up_acc0[r];
                        if (gate_bias) {
                            gate_value += *(const float *) ((const char *) gate_bias +
                                                            static_cast<int64_t>(expert_id) * gate_bias_nb1 +
                                                            static_cast<int64_t>(row) * sizeof(float));
                        }
                        if (up_bias) {
                            up_value += *(const float *) ((const char *) up_bias +
                                                          static_cast<int64_t>(expert_id) * up_bias_nb1 +
                                                          static_cast<int64_t>(row) * sizeof(float));
                        }
                        const float value = mmvq_moe_apply_pair_glu_esimd<GLU_OP>(gate_value, up_value, alpha, limit);
                        block_store<float, 1>(dst_out + row, value);
                    }
                }
                if (have_m1) {
#pragma unroll
                    for (int r = 0; r < Repeat; ++r) {
                        const int row = static_cast<int>(tile_m1) * Repeat + r;
                        if (row < nrows_per_expert) {
                            float gate_value = gate_acc1[r];
                            float up_value   = up_acc1[r];
                            if (gate_bias) {
                                gate_value += *(const float *) ((const char *) gate_bias +
                                                                static_cast<int64_t>(expert_id) * gate_bias_nb1 +
                                                                static_cast<int64_t>(row) * sizeof(float));
                            }
                            if (up_bias) {
                                up_value += *(const float *) ((const char *) up_bias +
                                                              static_cast<int64_t>(expert_id) * up_bias_nb1 +
                                                              static_cast<int64_t>(row) * sizeof(float));
                            }
                            const float value =
                                mmvq_moe_apply_pair_glu_esimd<GLU_OP>(gate_value, up_value, alpha, limit);
                            block_store<float, 1>(dst_out + row, value);
                        }
                    }
                }
            });
    });
    // clang-format on
}

#    line 1 "tools/sycl-kernel-bench/kernels/reference/mxfp4_pair_glu_source_line_device.cpp"

namespace ggml_sycl {

static bool mxfp4_pair_glu_source_line_accepts_default(const mxfp4_pair_glu_bench_args & args) {
    if (args.stream == nullptr || args.gate_ptrs == nullptr || args.up_ptrs == nullptr ||
        args.activations_q8_soa == nullptr || args.output == nullptr || args.dpas_b_packed == nullptr ||
        args.dpas_y_scales == nullptr || args.ids == nullptr) {
        return false;
    }
    if (args.gate_tmp != nullptr || args.up_tmp != nullptr || args.down_q8_soa != nullptr ||
        args.grouped_expert_ids != nullptr || args.grouped_offsets != nullptr || args.grouped_row_slots != nullptr ||
        args.grouped_chunks != nullptr || args.grouped_row_starts != nullptr) {
        return false;
    }
    if (args.ncols <= 0 || args.ncols_y <= 0 || args.nrows_per_expert <= 0 || args.num_experts <= 0 ||
        args.n_ids <= 0 || args.n_tokens <= 0 || args.ne11 <= 0 ||
        (args.ncols % GGML_SYCL_MXFP4_MOE_XMX_K_SOURCE_LINE) != 0) {
        return false;
    }
    if (args.ids_nb0 <= 0 || args.ids_nb1 <= 0 || args.nb11 <= 0 || args.nb12 <= 0 || args.dst_nb1 <= 0 ||
        args.dst_nb2 <= 0) {
        return false;
    }
    if ((args.gate_bias != nullptr && args.gate_bias_nb1 <= 0) || (args.up_bias != nullptr && args.up_bias_nb1 <= 0)) {
        return false;
    }

    const bool default_path = args.xmx_tiled && args.xmx_tiled_pack_q8 && args.rows_per_wg == 8 &&
                              args.xmx_tiled_m_tiles == 2 && args.xmx_tiles_n == 1 &&
                              args.glu_op == GGML_GLU_OP_SWIGLU_OAI && args.subgroup_size == 32;
    const bool unsupported_path =
        args.cache_y || args.direct_xmx || args.xmx_tiled_grouped || args.device_grouped || args.xmx_tiled_prefetch ||
        args.xmx_tiled_v2 || args.xmx_tiled_bundle4 || args.split_gate_up || args.single_column_gateup ||
        args.multi_rhs_gateup || args.predecoded_i8 || args.vector_qs_load || args.ignore_weight_scale ||
        args.scale_stride_blocks != 0 || args.grouped_n_chunks != 0 || args.down_q8_nb11 != 0 ||
        args.multi_rhs_cols != 1 || args.xmx_tiled_v2_group_bytes != 320 || args.xmx_tiled_bundle4_group_bytes != 0;
    return default_path && !unsupported_path;
}

bool ggml_sycl_mxfp4_pair_glu_bench_launch(const mxfp4_pair_glu_bench_args & args) {
    if (!mxfp4_pair_glu_source_line_accepts_default(args)) {
        return false;
    }

    sycl::queue & queue         = *args.stream;
    const int     total_batches = args.n_ids * args.n_tokens;
    sycl::event   pack_event    = mxfp4_dpas_pack_q8_source_line_sycl(
        queue, args.activations_q8_soa, args.dpas_b_packed, args.dpas_y_scales, args.ncols, args.ncols_y, total_batches,
        args.n_tokens, args.ne11, args.nb11, args.nb12);

    (void) mxfp4_pair_glu_xmx_tiled_dpas_m2_sycl<8, GGML_GLU_OP_SWIGLU_OAI, false, false>(
        queue, args.gate_ptrs, args.up_ptrs, args.dpas_b_packed, args.dpas_y_scales, args.output, args.ids,
        args.gate_bias, args.up_bias, args.ncols, args.nrows_per_expert, total_batches, args.n_tokens, args.ids_nb0,
        args.ids_nb1, args.dst_nb1, args.dst_nb2, args.gate_bias_nb1, args.up_bias_nb1, args.alpha, args.limit,
        GGML_SYCL_MXFP4_MOE_XMX_N_SOURCE_LINE, pack_event);
    return true;
}

}  // namespace ggml_sycl

#else
#    error "source-line MXFP4 device TU must only be compiled for sycl-mxfp4-source-line-probe"
#endif  // SYCL_MXFP4_SOURCE_LINE_PROBE_ONLY
