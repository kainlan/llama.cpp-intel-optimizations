#include "../dpas_exploration/dpas_kernels.hpp"
#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "reference_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <type_traits>
#include <vector>

namespace sycl_bench {

static inline int8_t mxfp4_code_value(uint8_t v);

static size_t sparse_expert_slot(size_t sel, size_t selected_count, size_t expert_slots) {
    if (selected_count <= 1) {
        return 0;
    }
    return (sel * (expert_slots - 1)) / (selected_count - 1);
}

static bool make_mxfp4_soa_scale_stride_layout(const std::vector<uint8_t> & src,
                                               int64_t                      m,
                                               int64_t                      k,
                                               int                          scale_stride_blocks,
                                               std::vector<uint8_t> &       dst,
                                               std::string &                error) {
    const int64_t blocks_per_row = k / QK_MXFP4;
    if (scale_stride_blocks < blocks_per_row) {
        error = "MXFP4 scale stride is smaller than blocks per row.";
        return false;
    }
    const size_t qs_bytes       = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row) * (QK_MXFP4 / 2);
    const size_t compact_scales = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    if (src.size() < qs_bytes + compact_scales) {
        error = "MXFP4 SOA source layout is too small for requested scale stride conversion.";
        return false;
    }
    if (scale_stride_blocks == blocks_per_row) {
        dst = src;
        return true;
    }

    const size_t padded_scales = static_cast<size_t>(m) * static_cast<size_t>(scale_stride_blocks);
    dst.assign(qs_bytes + padded_scales, 0);
    std::copy(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(qs_bytes), dst.begin());

    const uint8_t * src_scales = src.data() + qs_bytes;
    uint8_t *       dst_scales = dst.data() + qs_bytes;
    for (int64_t row = 0; row < m; ++row) {
        const size_t src_off = static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row);
        const size_t dst_off = static_cast<size_t>(row) * static_cast<size_t>(scale_stride_blocks);
        std::copy(src_scales + src_off, src_scales + src_off + blocks_per_row, dst_scales + dst_off);
    }
    return true;
}

static bool make_mxfp4_predecoded_i8_layout(const std::vector<uint8_t> & src,
                                            int64_t                      m,
                                            int64_t                      k,
                                            int                          scale_stride_blocks,
                                            std::vector<uint8_t> &       dst,
                                            std::string &                error) {
    const int64_t blocks_per_row = k / QK_MXFP4;
    if (scale_stride_blocks < blocks_per_row) {
        error = "MXFP4 predecoded scale stride is smaller than blocks per row.";
        return false;
    }

    const size_t qs_bytes    = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row) * (QK_MXFP4 / 2);
    const size_t scale_bytes = static_cast<size_t>(m) * static_cast<size_t>(scale_stride_blocks);
    if (src.size() < qs_bytes + scale_bytes) {
        error = "MXFP4 SOA source layout is too small for predecoded conversion.";
        return false;
    }

    const size_t i8_bytes = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row) * QK_MXFP4;
    dst.assign(i8_bytes + scale_bytes, 0);

    const uint8_t * src_qs     = src.data();
    const uint8_t * src_scales = src.data() + qs_bytes;
    uint8_t *       dst_i8     = dst.data();
    uint8_t *       dst_scales = dst.data() + i8_bytes;

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t * packed =
                src_qs + (static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(b)) *
                             (QK_MXFP4 / 2);
            int8_t * out =
                reinterpret_cast<int8_t *>(dst_i8) +
                (static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(b)) * QK_MXFP4;
            for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                out[i]                 = mxfp4_code_value(packed[i] & 0x0f);
                out[QK_MXFP4 / 2 + i] = mxfp4_code_value(packed[i] >> 4);
            }
        }

        const size_t scale_src = static_cast<size_t>(row) * static_cast<size_t>(scale_stride_blocks);
        std::copy(src_scales + scale_src, src_scales + scale_src + scale_stride_blocks, dst_scales + scale_src);
    }

    return true;
}

static bool make_mxfp4_xmx_tiled_layout(const std::vector<uint8_t> & src,
                                        int64_t                      m,
                                        int64_t                      k,
                                        int                          tile_n_total,
                                        std::vector<uint8_t> &       dst,
                                        std::string &                error) {
    if (m <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || tile_n_total <= 0 || (tile_n_total % 16) != 0) {
        error = "MXFP4 XMX-tiled layout requires positive block-aligned dimensions and a 16-column tile multiple.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const size_t  qs_bytes       = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row) * (QK_MXFP4 / 2);
    const size_t  scale_bytes    = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    if (src.size() < qs_bytes + scale_bytes) {
        error = "MXFP4 SOA source layout is too small for XMX-tiled conversion.";
        return false;
    }

    const int64_t output_tiles    = (m + tile_n_total - 1) / tile_n_total;
    const size_t  tile_group_size = static_cast<size_t>(tile_n_total) * (1 + QK_MXFP4 / 2);
    dst.assign(static_cast<size_t>(blocks_per_row) * static_cast<size_t>(output_tiles) * tile_group_size, 0);

    const uint8_t * src_qs     = src.data();
    const uint8_t * src_scales = src.data() + qs_bytes;

    for (int64_t k_block = 0; k_block < blocks_per_row; ++k_block) {
        for (int64_t tile = 0; tile < output_tiles; ++tile) {
            uint8_t * group = dst.data() + (static_cast<size_t>(k_block) * static_cast<size_t>(output_tiles) +
                                            static_cast<size_t>(tile)) *
                                               tile_group_size;
            uint8_t * scales = group;
            uint8_t * qs     = group + tile_n_total;
            for (int tn = 0; tn < tile_n_total; ++tn) {
                const int64_t row = tile * tile_n_total + tn;
                if (row >= m) {
                    continue;
                }
                const size_t block =
                    static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(k_block);
                scales[tn] = src_scales[block];
                std::copy(src_qs + block * (QK_MXFP4 / 2), src_qs + (block + 1) * (QK_MXFP4 / 2),
                          qs + static_cast<size_t>(tn) * (QK_MXFP4 / 2));
            }
        }
    }
    return true;
}

static int normalize_supported_xmx_tiles_n(int value) {
    if (value >= 4) {
        return 4;
    }
    if (value >= 2) {
        return 2;
    }
    return 1;
}

static bool select_mxfp4_xmx_tiles_n(sycl::queue & queue, int requested, int & tiles_n, std::string & error) {
    if (requested != 0 && requested != 1 && requested != 2 && requested != 4) {
        error = "requested XMX tile count must be 0, 1, 2, or 4.";
        return false;
    }

    const int    device_id = ggml_sycl_get_device_id_from_queue(queue);
    const auto & caps      = ggml_sycl_info().devices[device_id].xmx_caps;
    if (!xmx_capabilities_match_int8_tile(caps, GGML_SYCL_MXFP4_MOE_XMX_M, GGML_SYCL_MXFP4_MOE_XMX_N,
                                          GGML_SYCL_MXFP4_MOE_XMX_K)) {
        error = "device does not report the required int8 XMX tile shape for MXFP4 tiled decode.";
        return false;
    }
    if (!xmx_capabilities_support_sub_group(caps, GGML_SYCL_MXFP4_MOE_XMX_SG)) {
        error = "device does not report subgroup size 16 support for MXFP4 tiled decode.";
        return false;
    }

    tiles_n                  = requested != 0 ? requested : normalize_supported_xmx_tiles_n(caps.optimal_tiles_n);
    const auto slm_bytes_for = [](int candidate) {
        constexpr int xmx_m  = static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_M);
        constexpr int xmx_n  = static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N);
        constexpr int xmx_k  = static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_K);
        const int     tile_n = candidate * xmx_n;
        return static_cast<size_t>(xmx_m * xmx_k * sizeof(int8_t) + tile_n * xmx_k * sizeof(int8_t) +
                                   16 * sizeof(int8_t) + tile_n * sizeof(float) + sizeof(float));
    };

    while (tiles_n > 1 && caps.slm_size > 0 && slm_bytes_for(tiles_n) > caps.slm_size) {
        tiles_n /= 2;
    }
    if (caps.slm_size > 0 && slm_bytes_for(tiles_n) > caps.slm_size) {
        error = "device SLM is too small for the smallest compiled MXFP4 tiled decode variant.";
        return false;
    }
    return true;
}

static inline float mxfp4_e8m0_to_fp32_device(uint8_t x) {
    if (x == 0) {
        return sycl::bit_cast<float>(uint32_t{ 0x00400000 });
    }
    return sycl::bit_cast<float>(static_cast<uint32_t>(x) << 23);
}

static inline float mxfp4_value_device(uint8_t v) {
    switch (v & 0x0f) {
        case 0x0:
            return 0.0f;
        case 0x1:
            return 0.5f;
        case 0x2:
            return 1.0f;
        case 0x3:
            return 1.5f;
        case 0x4:
            return 2.0f;
        case 0x5:
            return 3.0f;
        case 0x6:
            return 4.0f;
        case 0x7:
            return 6.0f;
        case 0x8:
            return -0.0f;
        case 0x9:
            return -0.5f;
        case 0xa:
            return -1.0f;
        case 0xb:
            return -1.5f;
        case 0xc:
            return -2.0f;
        case 0xd:
            return -3.0f;
        case 0xe:
            return -4.0f;
        case 0xf:
            return -6.0f;
    }
    return 0.0f;
}

static inline int8_t mxfp4_code_value(uint8_t v) {
    static constexpr int8_t values[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    return values[v & 0x0f];
}

static inline float half_to_float_device(ggml_half v) {
    if constexpr (std::is_same_v<ggml_half, sycl::half>) {
        return static_cast<float>(v);
    } else {
        return static_cast<float>(sycl::bit_cast<sycl::half>(v));
    }
}

static bool validate_inline_dot(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                const std::vector<float> &   actual,
                                int64_t                      m,
                                int64_t                      n,
                                int64_t                      k,
                                std::string &                error) {
    std::vector<float> weight_row(static_cast<size_t>(k));
    std::string        dequant_error;
    const size_t       row_bytes = ggml_row_size(GGML_TYPE_MXFP4, k);

    double max_abs     = 0.0;
    double mean_abs    = 0.0;
    double max_ref_abs = 0.0;
    size_t count       = 0;

    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_ptr = weights.aos.data() + static_cast<size_t>(row) * row_bytes;
        if (!dequantize_row_fp32(GGML_TYPE_MXFP4, row_ptr, weight_row.data(), k, dequant_error)) {
            error = dequant_error;
            return false;
        }
        for (int64_t b = 0; b < n; ++b) {
            const ggml_half * src = activations.fp16.data() + static_cast<size_t>(b) * static_cast<size_t>(k);
            float             sum = 0.0f;
            for (int64_t kk = 0; kk < k; ++kk) {
                sum += weight_row[static_cast<size_t>(kk)] * bench_half_to_float(src[static_cast<size_t>(kk)]);
            }
            const size_t idx  = static_cast<size_t>(b) * static_cast<size_t>(m) + static_cast<size_t>(row);
            const double ref  = static_cast<double>(sum);
            const double diff = std::fabs(ref - static_cast<double>(actual[idx]));
            max_abs           = std::max(max_abs, diff);
            mean_abs += diff;
            max_ref_abs = std::max(max_ref_abs, std::fabs(ref));
            ++count;
        }
    }

    if (count > 0) {
        mean_abs /= static_cast<double>(count);
    }

    const double tol = 1.0e-3 + 2.0e-3 * max_ref_abs;
    if (max_abs > tol) {
        std::ostringstream oss;
        oss << "mxfp4_inline_dot validation failed: max_abs=" << max_abs << " mean_abs=" << mean_abs << " tol=" << tol;
        error = oss.str();
        return false;
    }
    return true;
}

bool run_mxfp4_inline_dot(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n,
                          int64_t                      k,
                          ggml_layout_mode             layout,
                          bool                         validate,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error) {
    if (m <= 0 || n <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "mxfp4_inline_dot requires positive M/N/K and K divisible by QK_MXFP4.";
        return false;
    }
    if (layout != GGML_LAYOUT_AOS && layout != GGML_LAYOUT_SOA) {
        error = "mxfp4_inline_dot supports AOS and SOA layouts only.";
        return false;
    }
    if (activations.fp16.empty()) {
        error = "mxfp4_inline_dot requires FP16 activations.";
        return false;
    }

    const std::vector<uint8_t> & host_weights = (layout == GGML_LAYOUT_AOS) ? weights.aos : weights.layout;
    if (host_weights.empty()) {
        error = "mxfp4_inline_dot received empty weight buffer.";
        return false;
    }

    const size_t  weight_bytes   = host_weights.size();
    const size_t  act_count      = activations.fp16.size();
    const size_t  act_bytes      = act_count * sizeof(ggml_half);
    const size_t  out_count      = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t  out_bytes      = out_count * sizeof(float);
    const int64_t blocks_per_row = k / QK_MXFP4;
    const size_t  nblocks        = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    constexpr int local_size     = 256;

    uint8_t *   d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    ggml_half * d_act     = sycl::malloc_device<ggml_half>(act_count, queue);
    float *     d_out     = sycl::malloc_device<float>(out_count, queue);

    if (!d_weights || !d_act || !d_out) {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        error = "device allocation failed for mxfp4_inline_dot.";
        return false;
    }

    queue.memcpy(d_weights, host_weights.data(), weight_bytes);
    queue.memcpy(d_act, activations.fp16.data(), act_bytes);
    queue.wait_and_throw();

    auto submit_dot = [&]() {
        return queue.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> partial(sycl::range<1>(local_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(out_count, local_size), sycl::range<2>(1, local_size)),
                [=](sycl::nd_item<2> item) {
                    const size_t task = item.get_group(0);
                    const size_t b    = task / static_cast<size_t>(m);
                    const size_t row  = task - b * static_cast<size_t>(m);
                    const size_t lid  = item.get_local_id(1);

                    float sum = 0.0f;
                    for (int64_t block = static_cast<int64_t>(lid); block < blocks_per_row; block += local_size) {
                        const size_t block_idx = row * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(block);
                        uint8_t      e         = 0;
                        const uint8_t * qs     = nullptr;

                        if (layout == GGML_LAYOUT_AOS) {
                            const size_t block_offset = block_idx * sizeof(block_mxfp4);
                            e                         = d_weights[block_offset];
                            qs                        = d_weights + block_offset + 1;
                        } else {
                            const uint8_t * qs_base = d_weights;
                            const uint8_t * e_base  = d_weights + nblocks * (QK_MXFP4 / 2);
                            e                       = e_base[block_idx];
                            qs                      = qs_base + block_idx * (QK_MXFP4 / 2);
                        }

                        const float  scale    = mxfp4_e8m0_to_fp32_device(e);
                        const size_t act_base = b * static_cast<size_t>(k) + static_cast<size_t>(block) * QK_MXFP4;
                        for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                            const uint8_t packed = qs[i];
                            const float   w0     = scale * mxfp4_value_device(packed & 0x0f);
                            const float   w1     = scale * mxfp4_value_device(packed >> 4);
                            sum += w0 * half_to_float_device(d_act[act_base + static_cast<size_t>(i)]);
                            sum += w1 * half_to_float_device(d_act[act_base + static_cast<size_t>(i + QK_MXFP4 / 2)]);
                        }
                    }

                    partial[lid] = sum;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = local_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            partial[lid] += partial[lid + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        d_out[task] = partial[0];
                    }
                });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_dot();
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit_dot();
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops      = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double bytes = static_cast<double>(weight_bytes) * static_cast<double>(n) + static_cast<double>(act_bytes) +
                         static_cast<double>(out_bytes);

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        if (!validate_inline_dot(weights, activations, actual, m, n, k, error)) {
            sycl::free(d_weights, queue);
            sycl::free(d_act, queue);
            sycl::free(d_out, queue);
            return false;
        }
    }

    sycl::free(d_weights, queue);
    sycl::free(d_act, queue);
    sycl::free(d_out, queue);
    return true;
}

bool run_mxfp4_selected_read(const GeneratedWeights & weights,
                             int64_t                  m,
                             int64_t                  n_selected,
                             int64_t                  k,
                             ggml_layout_mode         layout,
                             bool                     interleave_rows,
                             bool                     validate,
                             int                      warmup,
                             int                      iterations,
                             sycl::queue &            queue,
                             ReferenceMetrics &       out,
                             std::string &            error) {
    if (m <= 0 || n_selected <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "mxfp4_selected_read requires positive M/selected/K and K divisible by QK_MXFP4.";
        return false;
    }
    if (layout != GGML_LAYOUT_AOS && layout != GGML_LAYOUT_SOA) {
        error = "mxfp4_selected_read supports AOS and SOA layouts only.";
        return false;
    }

    const std::vector<uint8_t> & host_weights = (layout == GGML_LAYOUT_AOS) ? weights.aos : weights.layout;
    if (host_weights.empty()) {
        error = "mxfp4_selected_read received empty weight buffer.";
        return false;
    }

    const size_t  expert_bytes   = host_weights.size();
    const size_t  selected_count = static_cast<size_t>(n_selected);
    const size_t  weight_bytes   = expert_bytes * selected_count;
    const size_t  out_count      = static_cast<size_t>(m) * selected_count;
    const size_t  out_bytes      = out_count * sizeof(float);
    const int64_t blocks_per_row = k / QK_MXFP4;
    const size_t  nblocks        = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    constexpr int local_size     = 256;

    std::vector<uint8_t> host_weight_slices(weight_bytes);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        std::copy(host_weights.begin(), host_weights.end(), host_weight_slices.begin() + sel * expert_bytes);
    }

    uint8_t *        d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_ptrs    = sycl::malloc_device<const uint8_t *>(selected_count, queue);
    float *          d_out     = sycl::malloc_device<float>(out_count, queue);

    if (!d_weights || !d_ptrs || !d_out) {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_ptrs) {
            sycl::free(d_ptrs, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        error = "device allocation failed for mxfp4_selected_read.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(selected_count);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        host_ptrs[sel] = d_weights + sel * expert_bytes;
    }

    queue.memcpy(d_weights, host_weight_slices.data(), weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), selected_count * sizeof(const uint8_t *));
    queue.wait_and_throw();

    auto submit_read = [&]() {
        return queue.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> partial(sycl::range<1>(local_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(out_count, local_size), sycl::range<2>(1, local_size)),
                [=](sycl::nd_item<2> item) {
                    const size_t task = item.get_group(0);
                    const size_t sel  = interleave_rows ? task % selected_count : task / static_cast<size_t>(m);
                    const size_t row  = interleave_rows ? task / selected_count : task - sel * static_cast<size_t>(m);
                    const size_t lid  = item.get_local_id(1);

                    const uint8_t * expert = d_ptrs[sel];
                    float           sum    = 0.0f;
                    for (int64_t block = static_cast<int64_t>(lid); block < blocks_per_row; block += local_size) {
                        const size_t block_idx = row * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(block);
                        uint8_t      e         = 0;
                        const uint8_t * qs     = nullptr;

                        if (layout == GGML_LAYOUT_AOS) {
                            const size_t block_offset = block_idx * sizeof(block_mxfp4);
                            e                         = expert[block_offset];
                            qs                        = expert + block_offset + 1;
                        } else {
                            const uint8_t * qs_base = expert;
                            const uint8_t * e_base  = expert + nblocks * (QK_MXFP4 / 2);
                            e                       = e_base[block_idx];
                            qs                      = qs_base + block_idx * (QK_MXFP4 / 2);
                        }

                        float local_sum = static_cast<float>(e);
                        for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                            local_sum += static_cast<float>(qs[i]);
                        }
                        sum += local_sum;
                    }

                    partial[lid] = sum;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = local_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            partial[lid] += partial[lid + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        d_out[task] = partial[0];
                    }
                });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_read();
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit_read();
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double bytes    = static_cast<double>(weight_bytes) +
                         static_cast<double>(selected_count * sizeof(const uint8_t *)) + static_cast<double>(out_bytes);

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        for (float value : actual) {
            if (!std::isfinite(value) || value <= 0.0f) {
                error = "mxfp4_selected_read validation failed: non-positive or non-finite checksum.";
                sycl::free(d_weights, queue);
                sycl::free(d_ptrs, queue);
                sycl::free(d_out, queue);
                return false;
            }
        }
    }

    sycl::free(d_weights, queue);
    sycl::free(d_ptrs, queue);
    sycl::free(d_out, queue);
    return true;
}

bool run_mxfp4_pair_glu(const GeneratedWeights &     weights,
                        const GeneratedActivations & activations,
                        int64_t                      m,
                        int64_t                      n_selected,
                        int64_t                      k,
                        int64_t                      n_tokens,
                        int                          rows_per_wg,
                        bool                         cache_y,
                        bool                         direct_xmx,
                        bool                         split_gate_up,
                        int                          xmx_tiles_n,
                        bool                         vector_qs_load,
                        bool                         ignore_weight_scale,
                        int                          scale_stride_blocks,
                        int                          subgroup_size,
                        bool                         sparse_expert_slots,
                        bool                         use_bias,
                        bool                         validate,
                        int                          warmup,
                        int                          iterations,
                        sycl::queue &                queue,
                        ReferenceMetrics &           out,
                        std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_tokens <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0) {
        error = "mxfp4_pair_glu requires positive M/selected/tokens/K and K divisible by QK_MXFP4/QK8_1.";
        return false;
    }
    if (rows_per_wg != 1 && rows_per_wg != 2 && rows_per_wg != 4 && rows_per_wg != 8 && rows_per_wg != 16) {
        error = "mxfp4_pair_glu rows_per_wg must be one of 1, 2, 4, 8, or 16.";
        return false;
    }
    if (subgroup_size != 16 && subgroup_size != 32) {
        error = "mxfp4_pair_glu subgroup_size must be 16 or 32.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_pair_glu requires SOA MXFP4 weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_pair_glu requires SOA Q8_1 activations.";
        return false;
    }
    const int blocks_per_row = static_cast<int>(k / QK_MXFP4);
    const int scale_stride   = scale_stride_blocks > 0 ? scale_stride_blocks : blocks_per_row;
    if (scale_stride < blocks_per_row) {
        error = "mxfp4_pair_glu scale stride must be at least K / QK_MXFP4.";
        return false;
    }
    if (direct_xmx && (vector_qs_load || scale_stride_blocks > 0)) {
        error = "mxfp4_pair_glu direct-XMX diagnostic path does not support SOA load variants.";
        return false;
    }
    if (direct_xmx && xmx_tiles_n != 1 && xmx_tiles_n != 2 && xmx_tiles_n != 4) {
        error = "mxfp4_pair_glu direct-XMX tile count must be 1, 2, or 4.";
        return false;
    }

    const size_t         selected_count = static_cast<size_t>(n_selected);
    const size_t         token_count    = static_cast<size_t>(n_tokens);
    const size_t         ids_count      = selected_count * token_count;
    std::vector<uint8_t> expert_layout;
    if (!make_mxfp4_soa_scale_stride_layout(weights.layout, m, k, scale_stride, expert_layout, error)) {
        return false;
    }
    const size_t  expert_bytes         = expert_layout.size();
    const size_t  logical_expert_bytes = weights.layout.size();
    const size_t  expert_slots         = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  weight_bytes         = expert_bytes * expert_slots;
    const size_t  act_bytes            = activations.q8_1.size();
    const size_t  out_count            = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes            = out_count * sizeof(float);
    const size_t  bias_count           = static_cast<size_t>(m) * expert_slots;
    const size_t  bias_bytes           = bias_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);

    std::vector<uint8_t> gate_slices(weight_bytes);
    std::vector<uint8_t> up_slices(weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(expert_layout.begin(), expert_layout.end(), gate_slices.begin() + slot * expert_bytes);
        std::copy(expert_layout.begin(), expert_layout.end(), up_slices.begin() + slot * expert_bytes);
    }

    uint8_t *        d_gate       = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t *        d_up         = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_gate_ptrs  = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_up_ptrs    = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_act        = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids        = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out        = sycl::malloc_device<float>(out_count, queue);
    float *          d_gate_bias  = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    float *          d_up_bias    = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    const bool       need_ref_out = validate && (direct_xmx || split_gate_up || vector_qs_load);
    float *          d_ref_out    = need_ref_out ? sycl::malloc_device<float>(out_count, queue) : nullptr;
    float *          d_gate_tmp   = split_gate_up ? sycl::malloc_device<float>(out_count, queue) : nullptr;
    float *          d_up_tmp     = split_gate_up ? sycl::malloc_device<float>(out_count, queue) : nullptr;

    auto cleanup = [&]() {
        if (d_gate) {
            sycl::free(d_gate, queue);
        }
        if (d_up) {
            sycl::free(d_up, queue);
        }
        if (d_gate_ptrs) {
            sycl::free(d_gate_ptrs, queue);
        }
        if (d_up_ptrs) {
            sycl::free(d_up_ptrs, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_ids) {
            sycl::free(d_ids, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        if (d_gate_bias) {
            sycl::free(d_gate_bias, queue);
        }
        if (d_up_bias) {
            sycl::free(d_up_bias, queue);
        }
        if (d_ref_out) {
            sycl::free(d_ref_out, queue);
        }
        if (d_gate_tmp) {
            sycl::free(d_gate_tmp, queue);
        }
        if (d_up_tmp) {
            sycl::free(d_up_tmp, queue);
        }
    };

    if (!d_gate || !d_up || !d_gate_ptrs || !d_up_ptrs || !d_act || !d_ids || !d_out ||
        (use_bias && (!d_gate_bias || !d_up_bias)) || (need_ref_out && !d_ref_out) ||
        (split_gate_up && (!d_gate_tmp || !d_up_tmp))) {
        cleanup();
        error = "device allocation failed for mxfp4_pair_glu.";
        return false;
    }

    std::vector<const uint8_t *> host_gate_ptrs(expert_slots);
    std::vector<const uint8_t *> host_up_ptrs(expert_slots);
    std::vector<int32_t>         host_ids(ids_count);
    std::vector<float>           host_gate_bias;
    std::vector<float>           host_up_bias;
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_gate_ptrs[slot] = d_gate + slot * expert_bytes;
        host_up_ptrs[slot]   = d_up + slot * expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    queue.memcpy(d_gate, gate_slices.data(), weight_bytes);
    queue.memcpy(d_up, up_slices.data(), weight_bytes);
    queue.memcpy(d_gate_ptrs, host_gate_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_up_ptrs, host_up_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    if (use_bias) {
        host_gate_bias.resize(bias_count);
        host_up_bias.resize(bias_count);
        for (size_t i = 0; i < bias_count; ++i) {
            host_gate_bias[i] = 0.001f * static_cast<float>((i % 17) - 8);
            host_up_bias[i]   = 0.001f * static_cast<float>((i % 23) - 11);
        }
        queue.memcpy(d_gate_bias, host_gate_bias.data(), bias_bytes);
        queue.memcpy(d_up_bias, host_up_bias.data(), bias_bytes);
    }
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    ggml_sycl::mxfp4_pair_glu_bench_args args{};
    args.stream              = &queue;
    args.gate_ptrs           = reinterpret_cast<const void * const *>(d_gate_ptrs);
    args.up_ptrs             = reinterpret_cast<const void * const *>(d_up_ptrs);
    args.activations_q8_soa  = d_act;
    args.output              = d_out;
    args.gate_tmp            = d_gate_tmp;
    args.up_tmp              = d_up_tmp;
    args.ids                 = d_ids;
    args.ncols               = static_cast<int>(k);
    args.ncols_y             = static_cast<int>(k);
    args.nrows_per_expert    = static_cast<int>(m);
    args.num_experts         = static_cast<int>(expert_slots);
    args.n_ids               = static_cast<int>(n_selected);
    args.n_tokens            = static_cast<int>(n_tokens);
    args.ne11                = 1;
    args.ids_nb0             = sizeof(int32_t);
    args.ids_nb1             = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.nb11                = q8_row_bytes;
    args.nb12                = q8_row_bytes;
    args.dst_nb1             = static_cast<int64_t>(m * sizeof(float));
    args.dst_nb2             = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.gate_bias           = d_gate_bias;
    args.up_bias             = d_up_bias;
    args.gate_bias_nb1       = static_cast<int64_t>(m * sizeof(float));
    args.up_bias_nb1         = static_cast<int64_t>(m * sizeof(float));
    args.rows_per_wg         = rows_per_wg;
    args.cache_y             = cache_y;
    args.direct_xmx          = direct_xmx;
    args.split_gate_up       = split_gate_up;
    args.xmx_tiles_n         = xmx_tiles_n;
    args.vector_qs_load      = vector_qs_load;
    args.ignore_weight_scale = ignore_weight_scale;
    args.scale_stride_blocks = scale_stride_blocks;
    args.subgroup_size       = subgroup_size;
    args.glu_op              = GGML_GLU_OP_SWIGLU_OAI;
    args.alpha               = 1.702f;
    args.limit               = 7.0f;

    auto launch = [&]() {
        if (!ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch(args)) {
            error = "mxfp4_pair_glu launch rejected.";
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops = 4.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) *
                       static_cast<double>(k);
    const double selected_weight_bytes =
        static_cast<double>(logical_expert_bytes) * static_cast<double>(selected_count);
    const double bytes =
        2.0 * selected_weight_bytes * static_cast<double>(n_tokens) +
        static_cast<double>(split_gate_up ? 2 : 1) * static_cast<double>(act_bytes) + static_cast<double>(out_bytes) +
        (split_gate_up ? 4.0 * static_cast<double>(out_bytes) : 0.0) +
        (use_bias ? 2.0 * static_cast<double>(selected_count * static_cast<size_t>(m) * sizeof(float)) : 0.0) +
        2.0 * static_cast<double>(selected_count * sizeof(const uint8_t *)) +
        static_cast<double>(ids_count * sizeof(int32_t));

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        for (float value : actual) {
            if (!std::isfinite(value)) {
                cleanup();
                error = "mxfp4_pair_glu validation failed: non-finite output.";
                return false;
            }
        }
        if (need_ref_out) {
            queue.memset(d_ref_out, 0, out_bytes).wait();
            ggml_sycl::mxfp4_pair_glu_bench_args ref_args = args;
            ref_args.output                               = d_ref_out;
            ref_args.direct_xmx                           = false;
            ref_args.split_gate_up                        = false;
            ref_args.gate_tmp                             = nullptr;
            ref_args.up_tmp                               = nullptr;
            ref_args.xmx_tiles_n                          = 1;
            ref_args.vector_qs_load                       = false;
            ref_args.rows_per_wg                          = 1;
            ref_args.cache_y                              = false;
            if (!ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch(ref_args)) {
                cleanup();
                error = "mxfp4_pair_glu reference launch rejected.";
                return false;
            }
            queue.wait_and_throw();
            std::vector<float> expected(out_count);
            queue.memcpy(expected.data(), d_ref_out, out_bytes).wait();
            double max_err = 0.0;
            double sum_err = 0.0;
            for (size_t i = 0; i < out_count; ++i) {
                const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
                max_err           = std::max(max_err, diff);
                sum_err += diff;
                const double tol = 1e-2 + 1e-2 * std::abs(static_cast<double>(expected[i]));
                if (diff > tol) {
                    cleanup();
                    char msg[256];
                    std::snprintf(msg, sizeof(msg),
                                  "mxfp4_pair_glu validation failed at %zu: actual=%.6f expected=%.6f "
                                  "diff=%.6f tol=%.6f max=%.6f mean=%.6f.",
                                  i, static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff, tol,
                                  max_err, sum_err / static_cast<double>(i + 1));
                    error = msg;
                    return false;
                }
            }
        }
    }

    cleanup();
    return true;
}

bool run_mxfp4_layer_glu_down(const GeneratedWeights &     weights,
                              const GeneratedActivations & activations,
                              int64_t                      m,
                              int64_t                      n_selected,
                              int64_t                      k,
                              int64_t                      n_tokens,
                              int                          rows_per_wg,
                              bool                         cache_y,
                              bool                         vector_qs_load,
                              bool                         ignore_weight_scale,
                              int                          scale_stride_blocks,
                              int                          subgroup_size,
                              bool                         sparse_expert_slots,
                              bool                         use_bias,
                              bool                         validate,
                              int                          warmup,
                              int                          iterations,
                              sycl::queue &                queue,
                              ReferenceMetrics &           out,
                              std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_tokens <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0 ||
        (m % QK_MXFP4) != 0 || (m % QK8_1) != 0) {
        error = "mxfp4_layer_glu_down requires positive M/selected/tokens/K and block-aligned M/K.";
        return false;
    }
    if (m != k) {
        error = "mxfp4_layer_glu_down harness currently expects square expert matrices.";
        return false;
    }
    if (rows_per_wg != 1 && rows_per_wg != 2 && rows_per_wg != 4 && rows_per_wg != 8 && rows_per_wg != 16) {
        error = "mxfp4_layer_glu_down rows_per_wg must be one of 1, 2, 4, 8, or 16.";
        return false;
    }
    if (subgroup_size != 16 && subgroup_size != 32) {
        error = "mxfp4_layer_glu_down subgroup_size must be 16 or 32.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_layer_glu_down requires SOA MXFP4 weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_layer_glu_down requires SOA Q8_1 activations.";
        return false;
    }

    const int blocks_per_row = static_cast<int>(k / QK_MXFP4);
    const int scale_stride   = scale_stride_blocks > 0 ? scale_stride_blocks : blocks_per_row;
    if (scale_stride < blocks_per_row) {
        error = "mxfp4_layer_glu_down scale stride must be at least K / QK_MXFP4.";
        return false;
    }

    const size_t         selected_count = static_cast<size_t>(n_selected);
    const size_t         token_count    = static_cast<size_t>(n_tokens);
    const size_t         ids_count      = selected_count * token_count;
    std::vector<uint8_t> expert_layout;
    if (!make_mxfp4_soa_scale_stride_layout(weights.layout, m, k, scale_stride, expert_layout, error)) {
        return false;
    }
    const size_t  expert_bytes         = expert_layout.size();
    const size_t  logical_expert_bytes = weights.layout.size();
    const size_t  expert_slots         = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  weight_bytes         = expert_bytes * expert_slots;
    const size_t  act_bytes            = activations.q8_1.size();
    const size_t  glu_count            = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  glu_bytes            = glu_count * sizeof(float);
    const size_t  out_count            = glu_count;
    const size_t  out_bytes            = out_count * sizeof(float);
    const size_t  bias_count           = static_cast<size_t>(m) * expert_slots;
    const size_t  bias_bytes           = bias_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);
    const size_t  down_q8_bytes        = ids_count * static_cast<size_t>(q8_row_bytes);

    std::vector<uint8_t> gate_slices(weight_bytes);
    std::vector<uint8_t> up_slices(weight_bytes);
    std::vector<uint8_t> down_slices(weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(expert_layout.begin(), expert_layout.end(), gate_slices.begin() + slot * expert_bytes);
        std::copy(expert_layout.begin(), expert_layout.end(), up_slices.begin() + slot * expert_bytes);
        std::copy(expert_layout.begin(), expert_layout.end(), down_slices.begin() + slot * expert_bytes);
    }

    uint8_t *        d_gate      = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t *        d_up        = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t *        d_down      = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_gate_ptrs = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_up_ptrs   = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_down_ptrs = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_act       = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids       = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_glu       = sycl::malloc_device<float>(glu_count, queue);
    uint8_t *        d_down_q8   = sycl::malloc_device<uint8_t>(down_q8_bytes, queue);
    float *          d_out       = sycl::malloc_device<float>(out_count, queue);
    float *          d_gate_bias = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    float *          d_up_bias   = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;

    auto cleanup = [&]() {
        if (d_gate) {
            sycl::free(d_gate, queue);
        }
        if (d_up) {
            sycl::free(d_up, queue);
        }
        if (d_down) {
            sycl::free(d_down, queue);
        }
        if (d_gate_ptrs) {
            sycl::free(d_gate_ptrs, queue);
        }
        if (d_up_ptrs) {
            sycl::free(d_up_ptrs, queue);
        }
        if (d_down_ptrs) {
            sycl::free(d_down_ptrs, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_ids) {
            sycl::free(d_ids, queue);
        }
        if (d_glu) {
            sycl::free(d_glu, queue);
        }
        if (d_down_q8) {
            sycl::free(d_down_q8, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        if (d_gate_bias) {
            sycl::free(d_gate_bias, queue);
        }
        if (d_up_bias) {
            sycl::free(d_up_bias, queue);
        }
    };

    if (!d_gate || !d_up || !d_down || !d_gate_ptrs || !d_up_ptrs || !d_down_ptrs || !d_act || !d_ids || !d_glu ||
        !d_down_q8 || !d_out || (use_bias && (!d_gate_bias || !d_up_bias))) {
        cleanup();
        error = "device allocation failed for mxfp4_layer_glu_down.";
        return false;
    }

    std::vector<const uint8_t *> host_gate_ptrs(expert_slots);
    std::vector<const uint8_t *> host_up_ptrs(expert_slots);
    std::vector<const uint8_t *> host_down_ptrs(expert_slots);
    std::vector<int32_t>         host_ids(ids_count);
    std::vector<float>           host_gate_bias;
    std::vector<float>           host_up_bias;
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_gate_ptrs[slot] = d_gate + slot * expert_bytes;
        host_up_ptrs[slot]   = d_up + slot * expert_bytes;
        host_down_ptrs[slot] = d_down + slot * expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    queue.memcpy(d_gate, gate_slices.data(), weight_bytes);
    queue.memcpy(d_up, up_slices.data(), weight_bytes);
    queue.memcpy(d_down, down_slices.data(), weight_bytes);
    queue.memcpy(d_gate_ptrs, host_gate_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_up_ptrs, host_up_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_down_ptrs, host_down_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    if (use_bias) {
        host_gate_bias.resize(bias_count);
        host_up_bias.resize(bias_count);
        for (size_t i = 0; i < bias_count; ++i) {
            host_gate_bias[i] = 0.001f * static_cast<float>((i % 17) - 8);
            host_up_bias[i]   = 0.001f * static_cast<float>((i % 23) - 11);
        }
        queue.memcpy(d_gate_bias, host_gate_bias.data(), bias_bytes);
        queue.memcpy(d_up_bias, host_up_bias.data(), bias_bytes);
    }
    queue.memset(d_glu, 0, glu_bytes);
    queue.memset(d_down_q8, 0, down_q8_bytes);
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    ggml_sycl::mxfp4_layer_glu_down_bench_args args{};
    args.stream                 = &queue;
    args.gate_ptrs              = reinterpret_cast<const void * const *>(d_gate_ptrs);
    args.up_ptrs                = reinterpret_cast<const void * const *>(d_up_ptrs);
    args.down_ptrs              = reinterpret_cast<const void * const *>(d_down_ptrs);
    args.activations_q8_soa     = d_act;
    args.glu_f32                = d_glu;
    args.down_q8_soa            = d_down_q8;
    args.output                 = d_out;
    args.ids                    = d_ids;
    args.gate_bias              = d_gate_bias;
    args.up_bias                = d_up_bias;
    args.activation_cols        = static_cast<int>(k);
    args.intermediate_cols      = static_cast<int>(m);
    args.output_rows_per_expert = static_cast<int>(m);
    args.num_experts            = static_cast<int>(expert_slots);
    args.n_ids                  = static_cast<int>(n_selected);
    args.n_tokens               = static_cast<int>(n_tokens);
    args.ne11                   = 1;
    args.ids_nb0                = sizeof(int32_t);
    args.ids_nb1                = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.activation_nb11        = q8_row_bytes;
    args.activation_nb12        = q8_row_bytes;
    args.glu_nb1                = static_cast<int64_t>(m * sizeof(float));
    args.glu_nb2                = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.down_q8_nb11           = q8_row_bytes;
    args.down_q8_nb12           = static_cast<int64_t>(selected_count) * q8_row_bytes;
    args.output_nb1             = static_cast<int64_t>(m * sizeof(float));
    args.output_nb2             = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.gate_bias_nb1          = static_cast<int64_t>(m * sizeof(float));
    args.up_bias_nb1            = static_cast<int64_t>(m * sizeof(float));
    args.rows_per_wg            = rows_per_wg;
    args.cache_y                = cache_y;
    args.vector_qs_load         = vector_qs_load;
    args.ignore_weight_scale    = ignore_weight_scale;
    args.scale_stride_blocks    = scale_stride_blocks;
    args.subgroup_size          = subgroup_size;
    args.glu_op                 = GGML_GLU_OP_SWIGLU_OAI;
    args.alpha                  = 1.702f;
    args.limit                  = 7.0f;

    auto launch = [&]() {
        if (!ggml_sycl::ggml_sycl_mxfp4_layer_glu_down_bench_launch(args)) {
            error = "mxfp4_layer_glu_down launch rejected.";
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops = 6.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) *
                       static_cast<double>(k);
    const double selected_weight_bytes =
        static_cast<double>(logical_expert_bytes) * static_cast<double>(selected_count);
    const double bytes =
        3.0 * selected_weight_bytes * static_cast<double>(n_tokens) + static_cast<double>(act_bytes) +
        static_cast<double>(glu_bytes) + static_cast<double>(down_q8_bytes) + static_cast<double>(out_bytes) +
        (use_bias ? 2.0 * static_cast<double>(selected_count * static_cast<size_t>(m) * sizeof(float)) : 0.0) +
        3.0 * static_cast<double>(selected_count * sizeof(const uint8_t *)) +
        static_cast<double>(ids_count * sizeof(int32_t));

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        for (float value : actual) {
            if (!std::isfinite(value)) {
                cleanup();
                error = "mxfp4_layer_glu_down validation failed: non-finite output.";
                return false;
            }
        }
    }

    cleanup();
    return true;
}

bool run_mxfp4_mmv_id(const GeneratedWeights &     weights,
                      const GeneratedActivations & activations,
                      int64_t                      m,
                      int64_t                      n_selected,
                      int64_t                      k,
                      int64_t                      n_tokens,
                      int                          rows_per_wg,
                      bool                         cache_y,
                      bool                         predecoded_i8,
                      bool                         validate,
                      bool                         vector_qs_load,
                      bool                         ignore_weight_scale,
                      int                          scale_stride_blocks,
                      int                          subgroup_size,
                      bool                         sparse_expert_slots,
                      int                          warmup,
                      int                          iterations,
                      sycl::queue &                queue,
                      ReferenceMetrics &           out,
                      std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_tokens <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0) {
        error = "mxfp4_mmv_id requires positive M/selected/tokens/K and K divisible by QK_MXFP4/QK8_1.";
        return false;
    }
    if (rows_per_wg != 1 && rows_per_wg != 2 && rows_per_wg != 4 && rows_per_wg != 8 && rows_per_wg != 16) {
        error = "mxfp4_mmv_id rows_per_wg must be one of 1, 2, 4, 8, or 16.";
        return false;
    }
    if (subgroup_size != 16 && subgroup_size != 32) {
        error = "mxfp4_mmv_id subgroup_size must be 16 or 32.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_mmv_id requires SOA MXFP4 weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_mmv_id requires SOA Q8_1 activations.";
        return false;
    }
    const int blocks_per_row = static_cast<int>(k / QK_MXFP4);
    const int scale_stride   = scale_stride_blocks > 0 ? scale_stride_blocks : blocks_per_row;
    if (scale_stride < blocks_per_row) {
        error = "mxfp4_mmv_id scale stride must be at least K / QK_MXFP4.";
        return false;
    }

    const size_t         selected_count = static_cast<size_t>(n_selected);
    const size_t         token_count    = static_cast<size_t>(n_tokens);
    const size_t         ids_count      = selected_count * token_count;
    std::vector<uint8_t> expert_layout;
    if (!make_mxfp4_soa_scale_stride_layout(weights.layout, m, k, scale_stride, expert_layout, error)) {
        return false;
    }
    std::vector<uint8_t> launch_layout;
    if (predecoded_i8) {
        if (!make_mxfp4_predecoded_i8_layout(expert_layout, m, k, scale_stride, launch_layout, error)) {
            return false;
        }
    } else {
        launch_layout = expert_layout;
    }
    const size_t  expert_bytes         = launch_layout.size();
    const size_t  logical_expert_bytes = predecoded_i8 ? launch_layout.size() : weights.layout.size();
    const size_t  expert_slots         = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  weight_bytes         = expert_bytes * expert_slots;
    const size_t  act_bytes            = activations.q8_1.size();
    const size_t  out_count            = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes            = out_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);

    std::vector<uint8_t> weight_slices(weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(launch_layout.begin(), launch_layout.end(), weight_slices.begin() + slot * expert_bytes);
    }

    const bool needs_ref = validate && (vector_qs_load || predecoded_i8);
    uint8_t *        d_weights     = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_ptrs        = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_ref_weights = predecoded_i8 ? sycl::malloc_device<uint8_t>(expert_layout.size() * expert_slots, queue) : nullptr;
    const uint8_t ** d_ref_ptrs    = predecoded_i8 ? sycl::malloc_device<const uint8_t *>(expert_slots, queue) : nullptr;
    uint8_t *        d_act         = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids         = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out         = sycl::malloc_device<float>(out_count, queue);
    float *          d_ref_out     = needs_ref ? sycl::malloc_device<float>(out_count, queue) : nullptr;

    auto cleanup = [&]() {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_ptrs) {
            sycl::free(d_ptrs, queue);
        }
        if (d_ref_weights) {
            sycl::free(d_ref_weights, queue);
        }
        if (d_ref_ptrs) {
            sycl::free(d_ref_ptrs, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_ids) {
            sycl::free(d_ids, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        if (d_ref_out) {
            sycl::free(d_ref_out, queue);
        }
    };

    if (!d_weights || !d_ptrs || !d_act || !d_ids || !d_out || (needs_ref && !d_ref_out) ||
        (predecoded_i8 && (!d_ref_weights || !d_ref_ptrs))) {
        cleanup();
        error = "device allocation failed for mxfp4_mmv_id.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(expert_slots);
    std::vector<const uint8_t *> host_ref_ptrs(expert_slots);
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_ptrs[slot] = d_weights + slot * expert_bytes;
        if (predecoded_i8) {
            host_ref_ptrs[slot] = d_ref_weights + slot * expert_layout.size();
        }
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    queue.memcpy(d_weights, weight_slices.data(), weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    if (predecoded_i8) {
        std::vector<uint8_t> ref_weight_slices(expert_layout.size() * expert_slots);
        for (size_t slot = 0; slot < expert_slots; ++slot) {
            std::copy(expert_layout.begin(), expert_layout.end(), ref_weight_slices.begin() + slot * expert_layout.size());
        }
        queue.memcpy(d_ref_weights, ref_weight_slices.data(), ref_weight_slices.size());
        queue.memcpy(d_ref_ptrs, host_ref_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    }
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    ggml_sycl::mxfp4_mmv_id_bench_args args{};
    args.stream              = &queue;
    args.expert_ptrs         = reinterpret_cast<const void * const *>(d_ptrs);
    args.activations_q8_soa  = d_act;
    args.output              = d_out;
    args.ids                 = d_ids;
    args.ncols               = static_cast<int>(k);
    args.ncols_y             = static_cast<int>(k);
    args.nrows_per_expert    = static_cast<int>(m);
    args.num_experts         = static_cast<int>(expert_slots);
    args.n_ids               = static_cast<int>(n_selected);
    args.n_tokens            = static_cast<int>(n_tokens);
    args.ne11                = 1;
    args.ids_nb0             = sizeof(int32_t);
    args.ids_nb1             = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.nb11                = q8_row_bytes;
    args.nb12                = q8_row_bytes;
    args.dst_nb1             = static_cast<int64_t>(m * sizeof(float));
    args.dst_nb2             = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.rows_per_wg         = rows_per_wg;
    args.cache_y             = cache_y;
    args.vector_qs_load      = vector_qs_load;
    args.ignore_weight_scale = ignore_weight_scale;
    args.scale_stride_blocks = scale_stride_blocks;
    args.subgroup_size       = subgroup_size;

    auto launch = [&]() {
        const bool launched = predecoded_i8 ? ggml_sycl::ggml_sycl_mxfp4_mmv_id_predecoded_bench_launch(args) :
                                              ggml_sycl::ggml_sycl_mxfp4_mmv_id_bench_launch(args);
        if (!launched) {
            error = "mxfp4_mmv_id launch rejected.";
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) *
                       static_cast<double>(k);
    const double selected_weight_bytes =
        static_cast<double>(logical_expert_bytes) * static_cast<double>(selected_count);
    const double bytes = selected_weight_bytes * static_cast<double>(n_tokens) + static_cast<double>(act_bytes) +
                         static_cast<double>(out_bytes) +
                         static_cast<double>(selected_count * sizeof(const uint8_t *)) +
                         static_cast<double>(ids_count * sizeof(int32_t));

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        for (float value : actual) {
            if (!std::isfinite(value)) {
                cleanup();
                error = "mxfp4_mmv_id validation failed: non-finite output.";
                return false;
            }
        }
        if (needs_ref) {
            queue.memset(d_ref_out, 0, out_bytes).wait();
            ggml_sycl::mxfp4_mmv_id_bench_args ref_args = args;
            ref_args.output         = d_ref_out;
            ref_args.expert_ptrs    = predecoded_i8 ? reinterpret_cast<const void * const *>(d_ref_ptrs) : args.expert_ptrs;
            ref_args.vector_qs_load = false;
            ref_args.rows_per_wg    = 1;
            ref_args.cache_y        = false;
            if (!ggml_sycl::ggml_sycl_mxfp4_mmv_id_bench_launch(ref_args)) {
                cleanup();
                error = "mxfp4_mmv_id reference launch rejected.";
                return false;
            }
            queue.wait_and_throw();
            std::vector<float> expected(out_count);
            queue.memcpy(expected.data(), d_ref_out, out_bytes).wait();
            double max_err = 0.0;
            double sum_err = 0.0;
            for (size_t i = 0; i < out_count; ++i) {
                const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
                max_err           = std::max(max_err, diff);
                sum_err += diff;
                const double tol = 1e-2 + 1e-2 * std::abs(static_cast<double>(expected[i]));
                if (diff > tol) {
                    cleanup();
                    char msg[256];
                    std::snprintf(msg, sizeof(msg),
                                  "mxfp4_mmv_id validation failed at %zu: actual=%.6f expected=%.6f diff=%.6f "
                                  "tol=%.6f max=%.6f mean=%.6f.",
                                  i, static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff, tol,
                                  max_err, sum_err / static_cast<double>(i + 1));
                    error = msg;
                    return false;
                }
            }
        }
    }

    cleanup();
    return true;
}

#if DPAS_EXPLORATION_ESIMD_AVAILABLE
template <int Repeat, int NTileRepeats, bool PackedScales> struct mxfp4_dpas_scaled_predecoded_kernel;

template <int Repeat, int NTileRepeats, bool PackedScales>
static bool launch_mxfp4_dpas_scaled_predecoded_kernel(const int8_t * a_base,
                                                       const int8_t * b_base,
                                                       const float *  w_scales,
                                                       const float *  y_scales,
                                                       float *        c_base,
                                                       int64_t        m,
                                                       int64_t        n,
                                                       int64_t        k,
                                                       sycl::queue &  queue,
                                                       std::string &  error) {
    static_assert(NTileRepeats == 1 || NTileRepeats == 2 || NTileRepeats == 4);
    constexpr int ExecN = 16;
    constexpr int KPer  = 32;
    constexpr int AN    = Repeat * KPer;
    constexpr int BN    = KPer * ExecN;
    if (!a_base || !b_base || !w_scales || !y_scales || !c_base) {
        error = "mxfp4_dpas_scaled_predecoded buffers are null.";
        return false;
    }
    if ((m % Repeat) != 0 || (n % ExecN) != 0 || (k % KPer) != 0) {
        error = "mxfp4_dpas_scaled_predecoded dims must be multiples of repeat, 16, and 32.";
        return false;
    }

    const int64_t m_tiles       = m / Repeat;
    const int64_t n_tiles       = n / ExecN;
    const int64_t n_tile_groups = n_tiles / NTileRepeats;
    const int64_t k_tiles       = k / KPer;
    const int64_t tiles         = m_tiles * n_tile_groups;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mxfp4_dpas_scaled_predecoded_kernel<Repeat, NTileRepeats, PackedScales>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tiles)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL DPAS_NUM_REGS(128) {
                using namespace sycl::ext::intel::esimd;
                const int64_t tile_idx     = static_cast<int64_t>(item.get_global_id(0));
                const int64_t tile_m       = tile_idx / n_tile_groups;
                const int64_t tile_group_n = tile_idx - tile_m * n_tile_groups;
                const int64_t tile_n_base  = tile_group_n * NTileRepeats;

                simd<float, Repeat * ExecN> acc0   = 0.0f;
                simd<float, Repeat * ExecN> acc1   = 0.0f;
                simd<float, Repeat * ExecN> acc2   = 0.0f;
                simd<float, Repeat * ExecN> acc3   = 0.0f;
                const int8_t *              a_ptr  = a_base + (tile_m * k_tiles) * AN;
                const int8_t *              b_ptr0 = b_base + ((tile_n_base + 0) * k_tiles) * BN;
                const int8_t * b_ptr1 = b_base + ((tile_n_base + (NTileRepeats >= 2 ? 1 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr2 = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 2 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr3 = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 3 : 0)) * k_tiles) * BN;

                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    simd<int8_t, AN> a_vec =
                        dpas_block_load<int8_t, AN, DpasMemoryPattern::DIRECT_GLOBAL>(a_ptr, false);
                    simd<float, Repeat> w_scale_vec;
                    if constexpr (PackedScales) {
                        w_scale_vec = block_load<float, Repeat>(w_scales + (tile_m * k_tiles + kt) * Repeat);
                    } else {
#    pragma unroll
                        for (int r = 0; r < Repeat; ++r) {
                            w_scale_vec[r] = w_scales[(tile_m * Repeat + r) * k_tiles + kt];
                        }
                    }

                    simd<int8_t, BN> b_vec0 =
                        dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr0, false);
                    simd<int, Repeat * ExecN> part0 = 0;
                    part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(part0, b_vec0, a_vec);

                    simd<float, ExecN> y_scale_vec0;
                    if constexpr (PackedScales) {
                        y_scale_vec0 = block_load<float, ExecN>(y_scales + ((tile_n_base + 0) * k_tiles + kt) * ExecN);
                    } else {
#    pragma unroll
                        for (int col = 0; col < ExecN; ++col) {
                            y_scale_vec0[col] = y_scales[((tile_n_base + 0) * ExecN + col) * k_tiles + kt];
                        }
                    }
#    pragma unroll
                    for (int r = 0; r < Repeat; ++r) {
                        const float        w_scale                = w_scale_vec[r];
                        simd<int, ExecN>   row_i                  = part0.template select<ExecN, 1>(r * ExecN);
                        simd<float, ExecN> row                    = convert<float>(row_i) * (y_scale_vec0 * w_scale);
                        acc0.template select<ExecN, 1>(r * ExecN) = acc0.template select<ExecN, 1>(r * ExecN) + row;
                    }

                    if constexpr (NTileRepeats >= 2) {
                        simd<int8_t, BN> b_vec1 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr1, false);
                        simd<int, Repeat * ExecN> part1 = 0;
                        part1 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(part1, b_vec1, a_vec);

                        simd<float, ExecN> y_scale_vec1;
                        if constexpr (PackedScales) {
                            y_scale_vec1 =
                                block_load<float, ExecN>(y_scales + ((tile_n_base + 1) * k_tiles + kt) * ExecN);
                        } else {
#    pragma unroll
                            for (int col = 0; col < ExecN; ++col) {
                                y_scale_vec1[col] = y_scales[((tile_n_base + 1) * ExecN + col) * k_tiles + kt];
                            }
                        }
#    pragma unroll
                        for (int r = 0; r < Repeat; ++r) {
                            const float        w_scale = w_scale_vec[r];
                            simd<int, ExecN>   row_i   = part1.template select<ExecN, 1>(r * ExecN);
                            simd<float, ExecN> row     = convert<float>(row_i) * (y_scale_vec1 * w_scale);
                            acc1.template select<ExecN, 1>(r * ExecN) = acc1.template select<ExecN, 1>(r * ExecN) + row;
                        }
                    }

                    if constexpr (NTileRepeats >= 4) {
                        simd<int8_t, BN> b_vec2 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr2, false);
                        simd<int8_t, BN> b_vec3 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr3, false);
                        simd<int, Repeat * ExecN> part2 = 0;
                        simd<int, Repeat * ExecN> part3 = 0;
                        part2 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(part2, b_vec2, a_vec);
                        part3 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(part3, b_vec3, a_vec);

                        simd<float, ExecN> y_scale_vec2;
                        simd<float, ExecN> y_scale_vec3;
                        if constexpr (PackedScales) {
                            y_scale_vec2 =
                                block_load<float, ExecN>(y_scales + ((tile_n_base + 2) * k_tiles + kt) * ExecN);
                            y_scale_vec3 =
                                block_load<float, ExecN>(y_scales + ((tile_n_base + 3) * k_tiles + kt) * ExecN);
                        } else {
#    pragma unroll
                            for (int col = 0; col < ExecN; ++col) {
                                y_scale_vec2[col] = y_scales[((tile_n_base + 2) * ExecN + col) * k_tiles + kt];
                                y_scale_vec3[col] = y_scales[((tile_n_base + 3) * ExecN + col) * k_tiles + kt];
                            }
                        }
#    pragma unroll
                        for (int r = 0; r < Repeat; ++r) {
                            const float        w_scale = w_scale_vec[r];
                            simd<int, ExecN>   row_i2  = part2.template select<ExecN, 1>(r * ExecN);
                            simd<int, ExecN>   row_i3  = part3.template select<ExecN, 1>(r * ExecN);
                            simd<float, ExecN> row2    = convert<float>(row_i2) * (y_scale_vec2 * w_scale);
                            simd<float, ExecN> row3    = convert<float>(row_i3) * (y_scale_vec3 * w_scale);
                            acc2.template select<ExecN, 1>(r * ExecN) =
                                acc2.template select<ExecN, 1>(r * ExecN) + row2;
                            acc3.template select<ExecN, 1>(r * ExecN) =
                                acc3.template select<ExecN, 1>(r * ExecN) + row3;
                        }
                    }

                    a_ptr += AN;
                    b_ptr0 += BN;
                    b_ptr1 += BN;
                    b_ptr2 += BN;
                    b_ptr3 += BN;
                }

#    pragma unroll
                for (int r = 0; r < Repeat; ++r) {
                    float * out_ptr0 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 0) * ExecN;
                    block_store(out_ptr0, acc0.template select<ExecN, 1>(r * ExecN));
                    if constexpr (NTileRepeats >= 2) {
                        float * out_ptr1 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 1) * ExecN;
                        block_store(out_ptr1, acc1.template select<ExecN, 1>(r * ExecN));
                    }
                    if constexpr (NTileRepeats >= 4) {
                        float * out_ptr2 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 2) * ExecN;
                        float * out_ptr3 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 3) * ExecN;
                        block_store(out_ptr2, acc2.template select<ExecN, 1>(r * ExecN));
                        block_store(out_ptr3, acc3.template select<ExecN, 1>(r * ExecN));
                    }
                }
            });
    });
    ev.wait_and_throw();
    return true;
}
#else
template <int Repeat, int NTileRepeats, bool PackedScales>
static bool launch_mxfp4_dpas_scaled_predecoded_kernel(const int8_t *,
                                                       const int8_t *,
                                                       const float *,
                                                       const float *,
                                                       float *,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       sycl::queue &,
                                                       std::string & error) {
    (void) NTileRepeats;
    (void) PackedScales;
    error = "SYCL ESIMD unavailable; mxfp4_dpas_scaled_predecoded disabled.";
    return false;
}
#endif

#if DPAS_EXPLORATION_ESIMD_AVAILABLE
template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<int8_t, N>
mxfp4_code_values_esimd(sycl::ext::intel::esimd::simd<uint8_t, N> codes) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, N> base_mag = codes & uint8_t{ 7 };
    simd<uint8_t, N> mag      = base_mag;
    mag.merge(simd<uint8_t, N>(6), base_mag == uint8_t{ 5 });
    mag.merge(simd<uint8_t, N>(8), base_mag == uint8_t{ 6 });
    mag.merge(simd<uint8_t, N>(12), base_mag == uint8_t{ 7 });
    mag.merge(simd<uint8_t, N>(0), codes == uint8_t{ 8 });

    simd<int8_t, N> values = mag;
    simd<int8_t, N> neg    = -values;
    values.merge(neg, (codes & uint8_t{ 8 }) != uint8_t{ 0 });
    return values;
}

template <int Repeat, int NTileRepeats> struct mxfp4_dpas_compact_raw_kernel;

template <int Repeat, int NTileRepeats>
static bool launch_mxfp4_dpas_compact_raw_kernel(const uint8_t * compact_a,
                                                 const int8_t *  b_base,
                                                 int32_t *       c_base,
                                                 int64_t         m,
                                                 int64_t         n,
                                                 int64_t         k,
                                                 sycl::queue &   queue,
                                                 std::string &   error) {
    static_assert(NTileRepeats == 1 || NTileRepeats == 2 || NTileRepeats == 4);
    constexpr int ExecN         = 16;
    constexpr int KPer          = 32;
    constexpr int AN            = Repeat * KPer;
    constexpr int BN            = KPer * ExecN;
    constexpr int CompactABytes = Repeat * (KPer / 2);
    if (!compact_a || !b_base || !c_base) {
        error = "mxfp4_dpas_compact_raw buffers are null.";
        return false;
    }
    if ((m % Repeat) != 0 || (n % ExecN) != 0 || (k % KPer) != 0) {
        error = "mxfp4_dpas_compact_raw dims must be multiples of repeat, 16, and 32.";
        return false;
    }

    const int64_t m_tiles       = m / Repeat;
    const int64_t n_tiles       = n / ExecN;
    const int64_t n_tile_groups = n_tiles / NTileRepeats;
    const int64_t k_tiles       = k / KPer;
    const int64_t tiles         = m_tiles * n_tile_groups;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mxfp4_dpas_compact_raw_kernel<Repeat, NTileRepeats>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tiles)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL DPAS_NUM_REGS(128) {
                using namespace sycl::ext::intel::esimd;
                const int64_t tile_idx     = static_cast<int64_t>(item.get_global_id(0));
                const int64_t tile_m       = tile_idx / n_tile_groups;
                const int64_t tile_group_n = tile_idx - tile_m * n_tile_groups;
                const int64_t tile_n_base  = tile_group_n * NTileRepeats;

                simd<int, Repeat * ExecN> acc0  = 0;
                simd<int, Repeat * ExecN> acc1  = 0;
                simd<int, Repeat * ExecN> acc2  = 0;
                simd<int, Repeat * ExecN> acc3  = 0;
                const uint8_t *           a_ptr = compact_a + (tile_m * k_tiles) * CompactABytes;
                const int8_t *            b_ptr0 = b_base + ((tile_n_base + 0) * k_tiles) * BN;
                const int8_t * b_ptr1 = b_base + ((tile_n_base + (NTileRepeats >= 2 ? 1 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr2 = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 2 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr3 = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 3 : 0)) * k_tiles) * BN;

                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    simd<uint8_t, CompactABytes> packed =
                        dpas_block_load<uint8_t, CompactABytes, DpasMemoryPattern::DIRECT_GLOBAL>(a_ptr, false);
                    simd<int8_t, AN> a_vec;
#    pragma unroll
                    for (int r = 0; r < Repeat; ++r) {
                        simd<uint8_t, KPer / 2> row = packed.template select<KPer / 2, 1>(r * (KPer / 2));
                        a_vec.template select<KPer / 2, 1>(r * KPer) =
                            mxfp4_code_values_esimd<KPer / 2>(row & uint8_t{ 0x0f });
                        a_vec.template select<KPer / 2, 1>(r * KPer + KPer / 2) =
                            mxfp4_code_values_esimd<KPer / 2>(row >> 4);
                    }

                    simd<int8_t, BN> b_vec0 =
                        dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr0, false);
                    acc0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(acc0, b_vec0, a_vec);

                    if constexpr (NTileRepeats >= 2) {
                        simd<int8_t, BN> b_vec1 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr1, false);
                        acc1 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(acc1, b_vec1, a_vec);
                    }

                    if constexpr (NTileRepeats >= 4) {
                        simd<int8_t, BN> b_vec2 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr2, false);
                        simd<int8_t, BN> b_vec3 =
                            dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr3, false);
                        acc2 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(acc2, b_vec2, a_vec);
                        acc3 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(acc3, b_vec3, a_vec);
                    }

                    a_ptr += CompactABytes;
                    b_ptr0 += BN;
                    b_ptr1 += BN;
                    b_ptr2 += BN;
                    b_ptr3 += BN;
                }

#    pragma unroll
                for (int r = 0; r < Repeat; ++r) {
                    int32_t * out_ptr0 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 0) * ExecN;
                    block_store(out_ptr0, acc0.template select<ExecN, 1>(r * ExecN));
                    if constexpr (NTileRepeats >= 2) {
                        int32_t * out_ptr1 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 1) * ExecN;
                        block_store(out_ptr1, acc1.template select<ExecN, 1>(r * ExecN));
                    }
                    if constexpr (NTileRepeats >= 4) {
                        int32_t * out_ptr2 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 2) * ExecN;
                        int32_t * out_ptr3 = c_base + (tile_m * Repeat + r) * n + (tile_n_base + 3) * ExecN;
                        block_store(out_ptr2, acc2.template select<ExecN, 1>(r * ExecN));
                        block_store(out_ptr3, acc3.template select<ExecN, 1>(r * ExecN));
                    }
                }
            });
    });
    ev.wait_and_throw();
    return true;
}
#else
template <int Repeat, int NTileRepeats>
static bool launch_mxfp4_dpas_compact_raw_kernel(const uint8_t *,
                                                 const int8_t *,
                                                 int32_t *,
                                                 int64_t,
                                                 int64_t,
                                                 int64_t,
                                                 sycl::queue &,
                                                 std::string & error) {
    (void) Repeat;
    (void) NTileRepeats;
    error = "SYCL ESIMD unavailable; mxfp4_dpas_compact_raw disabled.";
    return false;
}
#endif

bool run_mxfp4_dpas_grouped_scaled_predecoded(const GeneratedWeights &     weights,
                                              const GeneratedActivations & activations,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      k,
                                              int                          n_tile_repeats,
                                              bool                         packed_scales,
                                              bool                         validate,
                                              int                          warmup,
                                              int                          iterations,
                                              sycl::queue &                queue,
                                              ReferenceMetrics &           out,
                                              std::string &                error) {
    constexpr int repeat = 8;
    constexpr int exec_n = 16;
    constexpr int k_per  = 32;

    if (m <= 0 || n <= 0 || k <= 0 || (m % repeat) != 0 || (n % exec_n) != 0 || (k % k_per) != 0) {
        error = "mxfp4_dpas_scaled_predecoded requires M multiple of 8, N multiple of 16, and K multiple of 32.";
        return false;
    }
    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4) {
        error = "mxfp4_dpas_scaled_predecoded n-tile repeats must be 1, 2, or 4.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_dpas_scaled_predecoded requires SOA MXFP4 weights as the source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_dpas_scaled_predecoded requires SOA Q8_1 activations.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const int64_t m_tiles        = m / repeat;
    const int64_t n_tiles        = n / exec_n;
    const int64_t k_tiles        = k / k_per;
    if ((n_tiles % n_tile_repeats) != 0) {
        error = "mxfp4_dpas_scaled_predecoded dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }
    const size_t nblocks      = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t qs_bytes     = nblocks * (QK_MXFP4 / 2);
    const size_t scale_bytes  = nblocks;
    const size_t q8_row_bytes = static_cast<size_t>(k / QK8_1) * sizeof(block_q8_1);
    if (weights.layout.size() < qs_bytes + scale_bytes) {
        error = "mxfp4_dpas_scaled_predecoded source weight layout is too small.";
        return false;
    }
    if (activations.q8_1.size() < static_cast<size_t>(n) * q8_row_bytes) {
        error = "mxfp4_dpas_scaled_predecoded activation buffer is too small.";
        return false;
    }

    const size_t a_tile_elems  = static_cast<size_t>(repeat) * static_cast<size_t>(k_per);
    const size_t b_tile_elems  = static_cast<size_t>(k_per) * static_cast<size_t>(exec_n);
    const size_t a_elems       = static_cast<size_t>(m_tiles) * static_cast<size_t>(k_tiles) * a_tile_elems;
    const size_t b_elems       = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t c_elems       = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t a_bytes       = a_elems * sizeof(int8_t);
    const size_t b_bytes       = b_elems * sizeof(int8_t);
    const size_t c_bytes       = c_elems * sizeof(float);
    const size_t w_scale_count = static_cast<size_t>(m) * static_cast<size_t>(k_tiles);
    const size_t y_scale_count = static_cast<size_t>(n) * static_cast<size_t>(k_tiles);
    const size_t w_scale_bytes = w_scale_count * sizeof(float);
    const size_t y_scale_bytes = y_scale_count * sizeof(float);

    std::vector<int8_t> a_host(a_elems, 0);
    std::vector<int8_t> b_host(b_elems, 0);
    std::vector<float>  w_scale_host(w_scale_count, 0.0f);
    std::vector<float>  y_scale_host(y_scale_count, 0.0f);
    std::vector<int8_t> weight_raw;
    std::vector<int8_t> act_raw;
    if (validate) {
        weight_raw.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0);
        act_raw.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
    }
    auto w_scale_index = [&](int64_t row, int64_t kt) -> size_t {
        if (packed_scales) {
            return (static_cast<size_t>(row / repeat) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) *
                       static_cast<size_t>(repeat) +
                   static_cast<size_t>(row % repeat);
        }
        return static_cast<size_t>(row) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt);
    };
    auto y_scale_index = [&](int64_t row, int64_t kt) -> size_t {
        if (packed_scales) {
            return (static_cast<size_t>(row / exec_n) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) *
                       static_cast<size_t>(exec_n) +
                   static_cast<size_t>(row % exec_n);
        }
        return static_cast<size_t>(row) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt);
    };

    const uint8_t * qs_base    = weights.layout.data();
    const uint8_t * scale_base = weights.layout.data() + qs_bytes;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const size_t block_idx =
                static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
            w_scale_host[w_scale_index(row, kt)] = mxfp4_e8m0_to_fp32_device(scale_base[block_idx]) * 0.5f;
        }
    }

    for (int64_t tile_m = 0; tile_m < m_tiles; ++tile_m) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const size_t tile_off =
                (static_cast<size_t>(tile_m) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * a_tile_elems;
            for (int r = 0; r < repeat; ++r) {
                const int64_t row = tile_m * repeat + r;
                const size_t  block_idx =
                    static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
                const uint8_t * block_qs = qs_base + block_idx * (QK_MXFP4 / 2);
                for (int kk = 0; kk < k_per; ++kk) {
                    const uint8_t packed = block_qs[static_cast<size_t>(kk % 16)];
                    const uint8_t nibble = kk < 16 ? (packed & 0x0f) : (packed >> 4);
                    const int8_t  value  = mxfp4_code_value(nibble);
                    const int64_t k_elem = kt * k_per + kk;
                    a_host[tile_off + static_cast<size_t>(r) * static_cast<size_t>(k_per) + static_cast<size_t>(kk)] =
                        value;
                    if (validate) {
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)] =
                            value;
                    }
                }
            }
        }
    }

    auto pack_vnni_tile = [&](int8_t * dst, const int8_t * src) {
        size_t idx = 0;
        for (int k0 = 0; k0 < k_per; k0 += 4) {
            for (int col = 0; col < exec_n; ++col) {
                for (int p = 0; p < 4; ++p) {
                    dst[idx++] = src[(k0 + p) * exec_n + col];
                }
            }
        }
    };

    std::vector<int8_t> b_tile(b_tile_elems, 0);
    for (int64_t row = 0; row < n; ++row) {
        const uint8_t * q8_row  = activations.q8_1.data() + static_cast<size_t>(row) * q8_row_bytes;
        const uint8_t * ds_base = q8_row + static_cast<size_t>(k);
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            sycl::half d{};
            std::memcpy(&d, ds_base + static_cast<size_t>(kt) * sizeof(sycl::half2), sizeof(sycl::half));
            y_scale_host[y_scale_index(row, kt)] = static_cast<float>(d);
        }
    }

    for (int64_t tile_n = 0; tile_n < n_tiles; ++tile_n) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            std::fill(b_tile.begin(), b_tile.end(), 0);
            for (int kk = 0; kk < k_per; ++kk) {
                const int64_t k_elem = kt * k_per + kk;
                for (int col = 0; col < exec_n; ++col) {
                    const int64_t   row    = tile_n * exec_n + col;
                    const uint8_t * q8_row = activations.q8_1.data() + static_cast<size_t>(row) * q8_row_bytes;
                    const int8_t    value  = reinterpret_cast<const int8_t *>(q8_row)[k_elem];
                    b_tile[static_cast<size_t>(kk) * exec_n + static_cast<size_t>(col)] = value;
                    if (validate) {
                        act_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)] =
                            value;
                    }
                }
            }
            const size_t tile_off =
                (static_cast<size_t>(tile_n) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * b_tile_elems;
            pack_vnni_tile(b_host.data() + tile_off, b_tile.data());
        }
    }

    int8_t * d_a       = sycl::malloc_device<int8_t>(a_elems, queue);
    int8_t * d_b       = sycl::malloc_device<int8_t>(b_elems, queue);
    float *  d_w_scale = sycl::malloc_device<float>(w_scale_count, queue);
    float *  d_y_scale = sycl::malloc_device<float>(y_scale_count, queue);
    float *  d_c       = sycl::malloc_device<float>(c_elems, queue);
    auto     cleanup   = [&]() {
        if (d_a) {
            sycl::free(d_a, queue);
        }
        if (d_b) {
            sycl::free(d_b, queue);
        }
        if (d_w_scale) {
            sycl::free(d_w_scale, queue);
        }
        if (d_y_scale) {
            sycl::free(d_y_scale, queue);
        }
        if (d_c) {
            sycl::free(d_c, queue);
        }
    };
    if (!d_a || !d_b || !d_w_scale || !d_y_scale || !d_c) {
        cleanup();
        error = "device allocation failed for mxfp4_dpas_scaled_predecoded.";
        return false;
    }

    queue.memcpy(d_a, a_host.data(), a_bytes);
    queue.memcpy(d_b, b_host.data(), b_bytes);
    queue.memcpy(d_w_scale, w_scale_host.data(), w_scale_bytes);
    queue.memcpy(d_y_scale, y_scale_host.data(), y_scale_bytes);
    queue.memset(d_c, 0, c_bytes);
    queue.wait_and_throw();

    auto launch = [&]() {
        switch (n_tile_repeats) {
            case 1:
                if (packed_scales) {
                    return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 1, true>(d_a, d_b, d_w_scale, d_y_scale,
                                                                                       d_c, m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 1, false>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
            case 2:
                if (packed_scales) {
                    return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 2, true>(d_a, d_b, d_w_scale, d_y_scale,
                                                                                       d_c, m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 2, false>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
            case 4:
                if (packed_scales) {
                    return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 4, true>(d_a, d_b, d_w_scale, d_y_scale,
                                                                                       d_c, m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_scaled_predecoded_kernel<repeat, 4, false>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
            default:
                error = "mxfp4_dpas_scaled_predecoded unsupported n-tile repeats.";
                return false;
        }
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = iterations > 0 ? total_us / static_cast<double>(iterations) : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops      = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double bytes    = static_cast<double>(a_bytes + b_bytes + w_scale_bytes + y_scale_bytes + c_bytes);
    out.total_us          = mean_us;
    out.gemm_us           = mean_us;
    out.tops              = mean_s > 0.0 ? (ops / mean_s) / 1.0e12 : 0.0;
    out.tflops            = out.tops;
    out.bandwidth_gbps    = mean_s > 0.0 ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(c_elems);
        queue.memcpy(actual.data(), d_c, c_bytes).wait();
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                float ref = 0.0f;
                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    int32_t dot = 0;
                    for (int kk = 0; kk < k_per; ++kk) {
                        const int64_t k_elem = kt * k_per + kk;
                        const int32_t w =
                            weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)];
                        const int32_t a =
                            act_raw[static_cast<size_t>(col) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)];
                        dot += w * a;
                    }
                    const float ws = w_scale_host[w_scale_index(row, kt)];
                    const float ys = y_scale_host[y_scale_index(col, kt)];
                    ref += static_cast<float>(dot) * ws * ys;
                }
                const size_t idx  = static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float  diff = std::fabs(actual[idx] - ref);
                const float  tol  = std::max(0.05f, std::fabs(ref) * 1.0e-3f);
                if (!std::isfinite(actual[idx]) || diff > tol) {
                    std::ostringstream oss;
                    oss << "mxfp4_dpas_scaled_predecoded validation failed at row=" << row << " col=" << col
                        << " actual=" << actual[idx] << " expected=" << ref << " diff=" << diff << " tol=" << tol;
                    error = oss.str();
                    cleanup();
                    return false;
                }
            }
        }
    }

    cleanup();
    return true;
}

bool run_mxfp4_dpas_grouped_raw(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                int64_t                      m,
                                int64_t                      n,
                                int64_t                      k,
                                int                          n_tile_repeats,
                                bool                         use_prefetch4,
                                bool                         validate,
                                int                          warmup,
                                int                          iterations,
                                sycl::queue &                queue,
                                ReferenceMetrics &           out,
                                std::string &                error) {
    constexpr int           repeat        = 8;
    constexpr int           exec_n        = 16;
    constexpr int           k_per         = 32;
    static constexpr int8_t mxfp4_lut[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

    if (m <= 0 || n <= 0 || k <= 0 || (m % repeat) != 0 || (n % exec_n) != 0 || (k % k_per) != 0) {
        error = "mxfp4_dpas_grouped_raw requires M multiple of 8, N multiple of 16, and K multiple of 32.";
        return false;
    }
    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4) {
        error = "mxfp4_dpas_grouped_raw n-tile repeats must be 1, 2, or 4.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_dpas_grouped_raw requires SOA MXFP4 weights as the source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_dpas_grouped_raw requires SOA Q8_1 activations.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const int64_t m_tiles        = m / repeat;
    const int64_t n_tiles        = n / exec_n;
    const int64_t k_tiles        = k / k_per;
    if ((n_tiles % n_tile_repeats) != 0) {
        error = "mxfp4_dpas_grouped_raw dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }

    const size_t nblocks      = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t qs_bytes     = nblocks * (QK_MXFP4 / 2);
    const size_t scale_bytes  = nblocks;
    const size_t q8_row_bytes = static_cast<size_t>(k / QK8_1) * sizeof(block_q8_1);
    if (weights.layout.size() < qs_bytes + scale_bytes) {
        error = "mxfp4_dpas_grouped_raw source weight layout is too small.";
        return false;
    }
    if (activations.q8_1.size() < static_cast<size_t>(n) * q8_row_bytes) {
        error = "mxfp4_dpas_grouped_raw activation buffer is too small.";
        return false;
    }

    const size_t a_tile_elems = static_cast<size_t>(repeat) * static_cast<size_t>(k_per);
    const size_t b_tile_elems = static_cast<size_t>(k_per) * static_cast<size_t>(exec_n);
    const size_t a_elems      = static_cast<size_t>(m_tiles) * static_cast<size_t>(k_tiles) * a_tile_elems;
    const size_t b_elems      = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t c_elems      = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t a_bytes      = a_elems * sizeof(int8_t);
    const size_t b_bytes      = b_elems * sizeof(int8_t);
    const size_t c_bytes      = c_elems * sizeof(int32_t);

    std::vector<int8_t> a_host(a_elems, 0);
    std::vector<int8_t> b_host(b_elems, 0);
    std::vector<int8_t> weight_raw;
    std::vector<int8_t> act_raw;
    if (validate) {
        weight_raw.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0);
        act_raw.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
    }

    const uint8_t * qs_base = weights.layout.data();
    for (int64_t tile_m = 0; tile_m < m_tiles; ++tile_m) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const size_t tile_off =
                (static_cast<size_t>(tile_m) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * a_tile_elems;
            for (int r = 0; r < repeat; ++r) {
                const int64_t row = tile_m * repeat + r;
                for (int kk = 0; kk < k_per; ++kk) {
                    const int64_t k_elem = kt * k_per + kk;
                    const int64_t block  = k_elem / QK_MXFP4;
                    const int     in_blk = static_cast<int>(k_elem - block * QK_MXFP4);
                    const size_t  block_idx =
                        static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(block);
                    const uint8_t packed = qs_base[block_idx * (QK_MXFP4 / 2) + static_cast<size_t>(in_blk % 16)];
                    const uint8_t nibble = in_blk < 16 ? (packed & 0x0f) : (packed >> 4);
                    const int8_t  value  = mxfp4_lut[nibble];
                    a_host[tile_off + static_cast<size_t>(r) * static_cast<size_t>(k_per) + static_cast<size_t>(kk)] =
                        value;
                    if (validate) {
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)] =
                            value;
                    }
                }
            }
        }
    }

    auto pack_vnni_tile = [&](int8_t * dst, const int8_t * src) {
        size_t idx = 0;
        for (int k0 = 0; k0 < k_per; k0 += 4) {
            for (int col = 0; col < exec_n; ++col) {
                for (int p = 0; p < 4; ++p) {
                    dst[idx++] = src[(k0 + p) * exec_n + col];
                }
            }
        }
    };

    std::vector<int8_t> b_tile(b_tile_elems, 0);
    for (int64_t tile_n = 0; tile_n < n_tiles; ++tile_n) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            std::fill(b_tile.begin(), b_tile.end(), 0);
            for (int kk = 0; kk < k_per; ++kk) {
                const int64_t k_elem = kt * k_per + kk;
                for (int col = 0; col < exec_n; ++col) {
                    const int64_t   row    = tile_n * exec_n + col;
                    const uint8_t * q8_row = activations.q8_1.data() + static_cast<size_t>(row) * q8_row_bytes;
                    const int8_t    value  = reinterpret_cast<const int8_t *>(q8_row)[k_elem];
                    b_tile[static_cast<size_t>(kk) * exec_n + static_cast<size_t>(col)] = value;
                    if (validate) {
                        act_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)] =
                            value;
                    }
                }
            }
            const size_t tile_off =
                (static_cast<size_t>(tile_n) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * b_tile_elems;
            pack_vnni_tile(b_host.data() + tile_off, b_tile.data());
        }
    }

    int8_t *  d_a     = sycl::malloc_device<int8_t>(a_elems, queue);
    int8_t *  d_b     = sycl::malloc_device<int8_t>(b_elems, queue);
    int32_t * d_c     = sycl::malloc_device<int32_t>(c_elems, queue);
    auto      cleanup = [&]() {
        if (d_a) {
            sycl::free(d_a, queue);
        }
        if (d_b) {
            sycl::free(d_b, queue);
        }
        if (d_c) {
            sycl::free(d_c, queue);
        }
    };
    if (!d_a || !d_b || !d_c) {
        cleanup();
        error = "device allocation failed for mxfp4_dpas_grouped_raw.";
        return false;
    }

    queue.memcpy(d_a, a_host.data(), a_bytes);
    queue.memcpy(d_b, b_host.data(), b_bytes);
    queue.memset(d_c, 0, c_bytes);
    queue.wait_and_throw();

    DpasBenchArgs args{};
    args.a              = d_a;
    args.b              = d_b;
    args.c              = d_c;
    args.m              = m;
    args.n              = n;
    args.k              = k;
    args.type_a         = DpasType::INT8;
    args.type_b         = DpasType::INT8;
    args.type_acc       = DpasAccType::INT32;
    args.memory_pattern = use_prefetch4 ? DpasMemoryPattern::LSC_PREFETCH_4 : DpasMemoryPattern::DIRECT_GLOBAL;
    args.grf_mode       = DpasGrfMode::GRF_128;
    args.repeat         = repeat;
    args.n_tile_repeats = n_tile_repeats;
    args.stream         = &queue;

    auto launch = [&]() {
        if (!run_dpas_sweep(args, nullptr, error)) {
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = iterations > 0 ? total_us / static_cast<double>(iterations) : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops      = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double bytes    = static_cast<double>(a_bytes + b_bytes + c_bytes);
    out.total_us          = mean_us;
    out.gemm_us           = mean_us;
    out.tops              = mean_s > 0.0 ? (ops / mean_s) / 1.0e12 : 0.0;
    out.tflops            = out.tops;
    out.bandwidth_gbps    = mean_s > 0.0 ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<int32_t> actual(c_elems);
        queue.memcpy(actual.data(), d_c, c_bytes).wait();
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                int32_t ref = 0;
                for (int64_t kk = 0; kk < k; ++kk) {
                    const int32_t w =
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(kk)];
                    const int32_t a =
                        act_raw[static_cast<size_t>(col) * static_cast<size_t>(k) + static_cast<size_t>(kk)];
                    ref += w * a;
                }
                const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
                if (actual[idx] != ref) {
                    std::ostringstream oss;
                    oss << "mxfp4_dpas_grouped_raw validation failed at row=" << row << " col=" << col
                        << " actual=" << actual[idx] << " expected=" << ref;
                    error = oss.str();
                    cleanup();
                    return false;
                }
            }
        }
    }

    cleanup();
    return true;
}

bool run_mxfp4_dpas_grouped_compact_raw(const GeneratedWeights &     weights,
                                        const GeneratedActivations & activations,
                                        int64_t                      m,
                                        int64_t                      n,
                                        int64_t                      k,
                                        int                          n_tile_repeats,
                                        bool                         validate,
                                        int                          warmup,
                                        int                          iterations,
                                        sycl::queue &                queue,
                                        ReferenceMetrics &           out,
                                        std::string &                error) {
    constexpr int repeat = 8;
    constexpr int exec_n = 16;
    constexpr int k_per  = 32;

    if (m <= 0 || n <= 0 || k <= 0 || (m % repeat) != 0 || (n % exec_n) != 0 || (k % k_per) != 0) {
        error = "mxfp4_dpas_compact_raw requires M multiple of 8, N multiple of 16, and K multiple of 32.";
        return false;
    }
    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4) {
        error = "mxfp4_dpas_compact_raw n-tile repeats must be 1, 2, or 4.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_dpas_compact_raw requires SOA MXFP4 weights as the source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_dpas_compact_raw requires SOA Q8_1 activations.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const int64_t m_tiles        = m / repeat;
    const int64_t n_tiles        = n / exec_n;
    const int64_t k_tiles        = k / k_per;
    if ((n_tiles % n_tile_repeats) != 0) {
        error = "mxfp4_dpas_compact_raw dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }

    const size_t nblocks      = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t qs_bytes     = nblocks * (QK_MXFP4 / 2);
    const size_t scale_bytes  = nblocks;
    const size_t q8_row_bytes = static_cast<size_t>(k / QK8_1) * sizeof(block_q8_1);
    if (weights.layout.size() < qs_bytes + scale_bytes) {
        error = "mxfp4_dpas_compact_raw source weight layout is too small.";
        return false;
    }
    if (activations.q8_1.size() < static_cast<size_t>(n) * q8_row_bytes) {
        error = "mxfp4_dpas_compact_raw activation buffer is too small.";
        return false;
    }

    const size_t a_tile_bytes = static_cast<size_t>(repeat) * static_cast<size_t>(k_per / 2);
    const size_t b_tile_elems = static_cast<size_t>(k_per) * static_cast<size_t>(exec_n);
    const size_t a_bytes      = static_cast<size_t>(m_tiles) * static_cast<size_t>(k_tiles) * a_tile_bytes;
    const size_t b_elems      = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t c_elems      = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t b_bytes      = b_elems * sizeof(int8_t);
    const size_t c_bytes      = c_elems * sizeof(int32_t);

    std::vector<uint8_t> a_host(a_bytes, 0);
    std::vector<int8_t>  b_host(b_elems, 0);
    std::vector<int8_t>  weight_raw;
    std::vector<int8_t>  act_raw;
    if (validate) {
        weight_raw.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0);
        act_raw.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
    }

    const uint8_t * qs_base = weights.layout.data();
    for (int64_t tile_m = 0; tile_m < m_tiles; ++tile_m) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            const size_t tile_off =
                (static_cast<size_t>(tile_m) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * a_tile_bytes;
            for (int r = 0; r < repeat; ++r) {
                const int64_t row = tile_m * repeat + r;
                const size_t  block_idx =
                    static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
                const uint8_t * block_qs = qs_base + block_idx * (QK_MXFP4 / 2);
                std::copy(block_qs, block_qs + (QK_MXFP4 / 2),
                          a_host.data() + tile_off + static_cast<size_t>(r) * (QK_MXFP4 / 2));
                if (validate) {
                    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                        const uint8_t packed = block_qs[i];
                        const int64_t k0     = kt * k_per + i;
                        const int64_t k1     = kt * k_per + (QK_MXFP4 / 2) + i;
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k0)] =
                            mxfp4_code_value(static_cast<uint8_t>(packed & 0x0f));
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k1)] =
                            mxfp4_code_value(static_cast<uint8_t>(packed >> 4));
                    }
                }
            }
        }
    }

    auto pack_vnni_tile = [&](int8_t * dst, const int8_t * src) {
        size_t idx = 0;
        for (int k0 = 0; k0 < k_per; k0 += 4) {
            for (int col = 0; col < exec_n; ++col) {
                for (int p = 0; p < 4; ++p) {
                    dst[idx++] = src[(k0 + p) * exec_n + col];
                }
            }
        }
    };

    std::vector<int8_t> b_tile(b_tile_elems, 0);
    for (int64_t tile_n = 0; tile_n < n_tiles; ++tile_n) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            std::fill(b_tile.begin(), b_tile.end(), 0);
            for (int kk = 0; kk < k_per; ++kk) {
                const int64_t k_elem = kt * k_per + kk;
                for (int col = 0; col < exec_n; ++col) {
                    const int64_t   row    = tile_n * exec_n + col;
                    const uint8_t * q8_row = activations.q8_1.data() + static_cast<size_t>(row) * q8_row_bytes;
                    const int8_t    value  = reinterpret_cast<const int8_t *>(q8_row)[k_elem];
                    b_tile[static_cast<size_t>(kk) * exec_n + static_cast<size_t>(col)] = value;
                    if (validate) {
                        act_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k_elem)] =
                            value;
                    }
                }
            }
            const size_t tile_off =
                (static_cast<size_t>(tile_n) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) * b_tile_elems;
            pack_vnni_tile(b_host.data() + tile_off, b_tile.data());
        }
    }

    uint8_t * d_a     = sycl::malloc_device<uint8_t>(a_bytes, queue);
    int8_t *  d_b     = sycl::malloc_device<int8_t>(b_elems, queue);
    int32_t * d_c     = sycl::malloc_device<int32_t>(c_elems, queue);
    auto      cleanup = [&]() {
        if (d_a) {
            sycl::free(d_a, queue);
        }
        if (d_b) {
            sycl::free(d_b, queue);
        }
        if (d_c) {
            sycl::free(d_c, queue);
        }
    };
    if (!d_a || !d_b || !d_c) {
        cleanup();
        error = "device allocation failed for mxfp4_dpas_compact_raw.";
        return false;
    }

    queue.memcpy(d_a, a_host.data(), a_bytes);
    queue.memcpy(d_b, b_host.data(), b_bytes);
    queue.memset(d_c, 0, c_bytes);
    queue.wait_and_throw();

    auto launch = [&]() {
        switch (n_tile_repeats) {
            case 1:
                return launch_mxfp4_dpas_compact_raw_kernel<repeat, 1>(d_a, d_b, d_c, m, n, k, queue, error);
            case 2:
                return launch_mxfp4_dpas_compact_raw_kernel<repeat, 2>(d_a, d_b, d_c, m, n, k, queue, error);
            case 4:
                return launch_mxfp4_dpas_compact_raw_kernel<repeat, 4>(d_a, d_b, d_c, m, n, k, queue, error);
            default:
                error = "mxfp4_dpas_compact_raw unsupported n-tile repeats.";
                return false;
        }
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = iterations > 0 ? total_us / static_cast<double>(iterations) : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops      = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double bytes    = static_cast<double>(a_bytes + b_bytes + c_bytes);
    out.total_us          = mean_us;
    out.gemm_us           = mean_us;
    out.tops              = mean_s > 0.0 ? (ops / mean_s) / 1.0e12 : 0.0;
    out.tflops            = out.tops;
    out.bandwidth_gbps    = mean_s > 0.0 ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<int32_t> actual(c_elems);
        queue.memcpy(actual.data(), d_c, c_bytes).wait();
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                int32_t ref = 0;
                for (int64_t kk = 0; kk < k; ++kk) {
                    const int32_t w =
                        weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(kk)];
                    const int32_t a =
                        act_raw[static_cast<size_t>(col) * static_cast<size_t>(k) + static_cast<size_t>(kk)];
                    ref += w * a;
                }
                const size_t idx = static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
                if (actual[idx] != ref) {
                    std::ostringstream oss;
                    oss << "mxfp4_dpas_compact_raw validation failed at row=" << row << " col=" << col
                        << " actual=" << actual[idx] << " expected=" << ref;
                    error = oss.str();
                    cleanup();
                    return false;
                }
            }
        }
    }

    cleanup();
    return true;
}

bool run_mxfp4_mmv_id_xmx_tiled(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                int64_t                      m,
                                int64_t                      n_selected,
                                int64_t                      k,
                                int64_t                      n_tokens,
                                int                          requested_tiles_n,
                                bool                         sparse_expert_slots,
                                bool                         raw_accum,
                                bool                         validate,
                                int                          warmup,
                                int                          iterations,
                                sycl::queue &                queue,
                                ReferenceMetrics &           out,
                                std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_tokens <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0) {
        error = "mxfp4_mmv_id_xmx_tiled requires positive M/selected/tokens/K and K divisible by QK_MXFP4/QK8_1.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_mmv_id_xmx_tiled requires SOA MXFP4 weights as its model-load source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_mmv_id_xmx_tiled requires SOA Q8_1 activations.";
        return false;
    }

    int tiles_n = 0;
    if (!select_mxfp4_xmx_tiles_n(queue, requested_tiles_n, tiles_n, error)) {
        return false;
    }
    const int tile_n_total = tiles_n * static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N);

    std::vector<uint8_t> tiled_layout;
    if (!make_mxfp4_xmx_tiled_layout(weights.layout, m, k, tile_n_total, tiled_layout, error)) {
        return false;
    }

    const size_t  selected_count       = static_cast<size_t>(n_selected);
    const size_t  token_count          = static_cast<size_t>(n_tokens);
    const size_t  ids_count            = selected_count * token_count;
    const size_t  tiled_expert_bytes   = tiled_layout.size();
    const size_t  logical_expert_bytes = weights.layout.size();
    const size_t  expert_slots         = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  tiled_weight_bytes   = tiled_expert_bytes * expert_slots;
    const size_t  act_bytes            = activations.q8_1.size();
    const size_t  out_count            = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes            = out_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);
    const bool    compare_to_soa       = validate && !raw_accum;

    std::vector<uint8_t> tiled_slices(tiled_weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(tiled_layout.begin(), tiled_layout.end(), tiled_slices.begin() + slot * tiled_expert_bytes);
    }

    uint8_t *        d_tiled = sycl::malloc_device<uint8_t>(tiled_weight_bytes, queue);
    const uint8_t ** d_ptrs  = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_act   = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids   = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out   = sycl::malloc_device<float>(out_count, queue);
    uint8_t *        d_ref_weights =
        compare_to_soa ? sycl::malloc_device<uint8_t>(logical_expert_bytes * expert_slots, queue) : nullptr;
    const uint8_t ** d_ref_ptrs = compare_to_soa ? sycl::malloc_device<const uint8_t *>(expert_slots, queue) : nullptr;
    float *          d_ref_out  = compare_to_soa ? sycl::malloc_device<float>(out_count, queue) : nullptr;

    auto cleanup = [&]() {
        if (d_tiled) {
            sycl::free(d_tiled, queue);
        }
        if (d_ptrs) {
            sycl::free(d_ptrs, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_ids) {
            sycl::free(d_ids, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        if (d_ref_weights) {
            sycl::free(d_ref_weights, queue);
        }
        if (d_ref_ptrs) {
            sycl::free(d_ref_ptrs, queue);
        }
        if (d_ref_out) {
            sycl::free(d_ref_out, queue);
        }
    };

    if (!d_tiled || !d_ptrs || !d_act || !d_ids || !d_out ||
        (compare_to_soa && (!d_ref_weights || !d_ref_ptrs || !d_ref_out))) {
        cleanup();
        error = "device allocation failed for mxfp4_mmv_id_xmx_tiled.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(expert_slots);
    std::vector<const uint8_t *> host_ref_ptrs;
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_ptrs[slot] = d_tiled + slot * tiled_expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    queue.memcpy(d_tiled, tiled_slices.data(), tiled_weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    queue.memset(d_out, 0, out_bytes);
    if (compare_to_soa) {
        std::vector<uint8_t> ref_slices(logical_expert_bytes * expert_slots);
        for (size_t slot = 0; slot < expert_slots; ++slot) {
            std::copy(weights.layout.begin(), weights.layout.end(), ref_slices.begin() + slot * logical_expert_bytes);
        }
        host_ref_ptrs.resize(expert_slots);
        for (size_t slot = 0; slot < expert_slots; ++slot) {
            host_ref_ptrs[slot] = d_ref_weights + slot * logical_expert_bytes;
        }
        queue.memcpy(d_ref_weights, ref_slices.data(), ref_slices.size());
        queue.memcpy(d_ref_ptrs, host_ref_ptrs.data(), expert_slots * sizeof(const uint8_t *));
        queue.memset(d_ref_out, 0, out_bytes);
    }
    queue.wait_and_throw();

    ggml_sycl::mxfp4_mmv_id_xmx_tiled_bench_args args{};
    args.stream             = &queue;
    args.expert_ptrs        = reinterpret_cast<const void * const *>(d_ptrs);
    args.activations_q8_soa = d_act;
    args.output             = d_out;
    args.ids                = d_ids;
    args.ncols              = static_cast<int>(k);
    args.ncols_y            = static_cast<int>(k);
    args.nrows_per_expert   = static_cast<int>(m);
    args.num_experts        = static_cast<int>(expert_slots);
    args.n_ids              = static_cast<int>(n_selected);
    args.n_tokens           = static_cast<int>(n_tokens);
    args.ne11               = 1;
    args.xmx_tiles_n        = tiles_n;
    args.raw_accum          = raw_accum;
    args.ids_nb0            = sizeof(int32_t);
    args.ids_nb1            = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.nb11               = q8_row_bytes;
    args.nb12               = q8_row_bytes;
    args.dst_nb1            = static_cast<int64_t>(m * sizeof(float));
    args.dst_nb2            = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));

    auto launch = [&]() {
        if (!ggml_sycl::ggml_sycl_mxfp4_mmv_id_xmx_tiled_bench_launch(args)) {
            error = "mxfp4_mmv_id_xmx_tiled launch rejected.";
            return false;
        }
        return true;
    };

    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!launch()) {
            cleanup();
            return false;
        }
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) *
                       static_cast<double>(k);
    const double selected_weight_bytes =
        static_cast<double>(logical_expert_bytes) * static_cast<double>(selected_count);
    const double bytes = selected_weight_bytes * static_cast<double>(n_tokens) + static_cast<double>(act_bytes) +
                         static_cast<double>(out_bytes) +
                         static_cast<double>(selected_count * sizeof(const uint8_t *)) +
                         static_cast<double>(ids_count * sizeof(int32_t));

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        for (float value : actual) {
            if (!std::isfinite(value)) {
                cleanup();
                error = "mxfp4_mmv_id_xmx_tiled validation failed: non-finite output.";
                return false;
            }
        }

        if (raw_accum) {
            cleanup();
            return true;
        }

        ggml_sycl::mxfp4_mmv_id_bench_args ref_args{};
        ref_args.stream             = &queue;
        ref_args.expert_ptrs        = reinterpret_cast<const void * const *>(d_ref_ptrs);
        ref_args.activations_q8_soa = d_act;
        ref_args.output             = d_ref_out;
        ref_args.ids                = d_ids;
        ref_args.ncols              = static_cast<int>(k);
        ref_args.ncols_y            = static_cast<int>(k);
        ref_args.nrows_per_expert   = static_cast<int>(m);
        ref_args.num_experts        = static_cast<int>(expert_slots);
        ref_args.n_ids              = static_cast<int>(n_selected);
        ref_args.n_tokens           = static_cast<int>(n_tokens);
        ref_args.ne11               = 1;
        ref_args.ids_nb0            = sizeof(int32_t);
        ref_args.ids_nb1            = static_cast<int64_t>(selected_count * sizeof(int32_t));
        ref_args.nb11               = q8_row_bytes;
        ref_args.nb12               = q8_row_bytes;
        ref_args.dst_nb1            = static_cast<int64_t>(m * sizeof(float));
        ref_args.dst_nb2            = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
        ref_args.rows_per_wg        = 1;
        ref_args.cache_y            = false;
        if (!ggml_sycl::ggml_sycl_mxfp4_mmv_id_bench_launch(ref_args)) {
            cleanup();
            error = "mxfp4_mmv_id_xmx_tiled reference launch rejected.";
            return false;
        }
        queue.wait_and_throw();

        std::vector<float> expected(out_count);
        queue.memcpy(expected.data(), d_ref_out, out_bytes).wait();
        double max_err = 0.0;
        double sum_err = 0.0;
        for (size_t i = 0; i < out_count; ++i) {
            const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
            max_err           = std::max(max_err, diff);
            sum_err += diff;
            const double tol = 2e-2 + 2e-2 * std::abs(static_cast<double>(expected[i]));
            if (diff > tol) {
                cleanup();
                char msg[320];
                std::snprintf(msg, sizeof(msg),
                              "mxfp4_mmv_id_xmx_tiled validation failed at %zu: actual=%.6f expected=%.6f "
                              "diff=%.6f tol=%.6f max=%.6f mean=%.6f tiles_n=%d.",
                              i, static_cast<double>(actual[i]), static_cast<double>(expected[i]), diff, tol, max_err,
                              sum_err / static_cast<double>(i + 1), tiles_n);
                error = msg;
                return false;
            }
        }
    }

    cleanup();
    return true;
}

}  // namespace sycl_bench
