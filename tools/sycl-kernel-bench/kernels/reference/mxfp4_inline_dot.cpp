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
                out[i]                = mxfp4_code_value(packed[i] & 0x0f);
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

static bool make_mxfp4_selected_kmajor_layout(const std::vector<uint8_t> & src,
                                              int64_t                      m,
                                              int64_t                      n_selected,
                                              int64_t                      k,
                                              std::vector<uint8_t> &       dst,
                                              std::string &                error) {
    if (m <= 0 || n_selected <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "MXFP4 selected K-major layout requires positive M/selected/K and K divisible by QK_MXFP4.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const size_t  nblocks        = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t  qs_bytes       = nblocks * (QK_MXFP4 / 2);
    const size_t  scale_bytes    = nblocks;
    if (src.size() < qs_bytes + scale_bytes) {
        error = "MXFP4 SOA source layout is too small for selected K-major conversion.";
        return false;
    }

    constexpr size_t roles        = 2;  // gate and up
    constexpr size_t record_bytes = sizeof(block_mxfp4);
    const size_t     selected     = static_cast<size_t>(n_selected);
    const size_t     rows         = static_cast<size_t>(m);
    const size_t     blocks       = static_cast<size_t>(blocks_per_row);
    dst.assign(blocks * selected * roles * rows * record_bytes, 0);

    const uint8_t * src_qs     = src.data();
    const uint8_t * src_scales = src.data() + qs_bytes;
    for (size_t block = 0; block < blocks; ++block) {
        for (size_t sel = 0; sel < selected; ++sel) {
            for (size_t role = 0; role < roles; ++role) {
                for (size_t row = 0; row < rows; ++row) {
                    const size_t src_block = row * blocks + block;
                    const size_t dst_slot  = (((block * selected + sel) * roles + role) * rows + row) * record_bytes;
                    uint8_t *    record    = dst.data() + dst_slot;
                    record[0]              = src_scales[src_block];
                    std::copy(src_qs + src_block * (QK_MXFP4 / 2), src_qs + (src_block + 1) * (QK_MXFP4 / 2),
                              record + 1);
                }
            }
        }
    }
    return true;
}

static inline float mxfp4_selected_kmajor_record_checksum(const uint8_t * record) {
    float sum = static_cast<float>(record[0]);
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        sum += static_cast<float>(record[1 + i]);
    }
    return sum;
}

static inline float mxfp4_selected_kmajor_dot_q8_1(const uint8_t * record, const int8_t * q8_qs, float q8_scale) {
    const float scale = mxfp4_e8m0_to_fp32_device(record[0]) * q8_scale;
    float       sum   = 0.0f;
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        const uint8_t packed = record[1 + i];
        sum += scale * mxfp4_value_device(packed & 0x0f) * static_cast<float>(q8_qs[static_cast<size_t>(i)]);
        sum +=
            scale * mxfp4_value_device(packed >> 4) * static_cast<float>(q8_qs[static_cast<size_t>(i + QK_MXFP4 / 2)]);
    }
    return sum;
}

static inline float mxfp4_selected_kmajor_q8_1_scale(const uint8_t * q8_row, int64_t k, int64_t block) {
    const uint8_t *   scale_ptr = q8_row + static_cast<size_t>(k) + static_cast<size_t>(block) * sizeof(sycl::half2);
    const sycl::half2 ds_vals   = *reinterpret_cast<const sycl::half2 *>(scale_ptr);
    return static_cast<float>(ds_vals.x());
}

static inline float mxfp4_selected_kmajor_oai_glu(float gate, float up) {
    constexpr float alpha        = 1.702f;
    constexpr float limit        = 7.0f;
    const float     gate_limited = sycl::fmin(gate, limit);
    const float     up_limited   = sycl::fmax(sycl::fmin(up, limit), -limit);
    return (gate_limited / (1.0f + sycl::native::exp(-gate_limited * alpha))) * (1.0f + up_limited);
}

static inline float mxfp4_e8m0_to_fp32_prod_host(uint8_t e) {
    uint32_t bits;
    if (e < 2) {
        bits = e == 0 ? 0x00000000u : 0x33800000u;
    } else {
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

static inline float half_to_float_device(ggml_half v) {
    if constexpr (std::is_same_v<ggml_half, sycl::half>) {
        return static_cast<float>(v);
    } else {
        return static_cast<float>(sycl::bit_cast<sycl::half>(v));
    }
}

static bool validate_q8_1_soa_from_f32_rows(const std::vector<float> &   src,
                                            const std::vector<uint8_t> & q8_soa,
                                            int64_t                      rows,
                                            int64_t                      cols,
                                            int64_t                      q8_row_bytes,
                                            std::string &                error) {
    if (rows <= 0 || cols <= 0 || (cols % QK8_1) != 0 || q8_row_bytes <= 0 ||
        src.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols) ||
        q8_soa.size() < static_cast<size_t>(rows) * static_cast<size_t>(q8_row_bytes)) {
        error = "Q8_1 SOA validation received invalid dimensions.";
        return false;
    }

    const int64_t blocks_per_row = cols / QK8_1;
    double        max_d_diff     = 0.0;
    double        max_sum_diff   = 0.0;
    for (int64_t row = 0; row < rows; ++row) {
        const float *   src_row = src.data() + static_cast<size_t>(row) * static_cast<size_t>(cols);
        const uint8_t * q8_row  = q8_soa.data() + static_cast<size_t>(row) * static_cast<size_t>(q8_row_bytes);

        for (int64_t block = 0; block < blocks_per_row; ++block) {
            const float * src_block = src_row + static_cast<size_t>(block) * QK8_1;
            float         amax      = 0.0f;
            float         sum       = 0.0f;
            for (int lane = 0; lane < QK8_1; ++lane) {
                const float value = src_block[lane];
                if (!std::isfinite(value)) {
                    error = "Q8_1 SOA validation failed: non-finite source value.";
                    return false;
                }
                amax = std::max(amax, std::fabs(value));
                sum += value;
            }

            float d = amax == 0.0f ? 1.0f : amax / 127.0f;
            for (int lane = 0; lane < QK8_1; ++lane) {
                const int expected = static_cast<int>(std::round(src_block[lane] / d));
                const int actual   = static_cast<int>(
                    reinterpret_cast<const int8_t *>(q8_row)[static_cast<size_t>(block) * QK8_1 + lane]);
                if (actual != expected) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg),
                                  "Q8_1 SOA validation failed at row=%lld block=%lld lane=%d: actual=%d expected=%d.",
                                  static_cast<long long>(row), static_cast<long long>(block), lane, actual, expected);
                    error = msg;
                    return false;
                }
            }

            d = amax == 0.0f ? 0.0f : d;
            const uint8_t * ds_bytes =
                q8_row + static_cast<size_t>(cols) + static_cast<size_t>(block) * sizeof(sycl::half2);
            sycl::half2 ds_vals;
            std::memcpy(&ds_vals, ds_bytes, sizeof(ds_vals));
            const float  actual_d   = static_cast<float>(ds_vals.x());
            const float  actual_sum = static_cast<float>(ds_vals.y());
            const float  expected_d = static_cast<float>(sycl::half(d));
            const float  expected_s = static_cast<float>(sycl::half(sum));
            const double d_diff     = std::fabs(static_cast<double>(actual_d) - static_cast<double>(expected_d));
            const double s_diff     = std::fabs(static_cast<double>(actual_sum) - static_cast<double>(expected_s));
            const double d_tol      = std::max(1.0e-6, 1.0e-3 * std::fabs(static_cast<double>(expected_d)));
            const double s_tol      = std::max(1.0e-3, 1.0e-3 * std::fabs(static_cast<double>(expected_s)));
            max_d_diff              = std::max(max_d_diff, d_diff);
            max_sum_diff            = std::max(max_sum_diff, s_diff);
            if (d_diff > d_tol || s_diff > s_tol) {
                char msg[320];
                std::snprintf(msg, sizeof(msg),
                              "Q8_1 SOA validation failed at row=%lld block=%lld: d actual=%.8g expected=%.8g, "
                              "sum actual=%.8g expected=%.8g, max_d=%.8g max_sum=%.8g.",
                              static_cast<long long>(row), static_cast<long long>(block), actual_d, expected_d,
                              actual_sum, expected_s, max_d_diff, max_sum_diff);
                error = msg;
                return false;
            }
        }
    }
    return true;
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

bool run_mxfp4_selected_kmajor(const GeneratedWeights &     weights,
                               const GeneratedActivations & activations,
                               int64_t                      m,
                               int64_t                      n_selected,
                               int64_t                      k,
                               bool                         pair_glu,
                               bool                         tile_read,
                               bool                         validate,
                               int                          warmup,
                               int                          iterations,
                               sycl::queue &                queue,
                               ReferenceMetrics &           out,
                               std::string &                error) {
    if (m <= 0 || n_selected <= 0 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0) {
        error = "mxfp4_selected_kmajor requires positive M/selected/K and K divisible by QK_MXFP4/QK8_1.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_selected_kmajor requires SOA MXFP4 source weights.";
        return false;
    }
    if (pair_glu && activations.q8_1.empty()) {
        error = "mxfp4_selected_kmajor_pair_glu requires SOA Q8_1 activations.";
        return false;
    }
    if (pair_glu && tile_read) {
        error = "mxfp4_selected_kmajor cannot combine pair_glu and tile_read modes.";
        return false;
    }

    std::vector<uint8_t> kmajor_layout;
    if (!make_mxfp4_selected_kmajor_layout(weights.layout, m, n_selected, k, kmajor_layout, error)) {
        return false;
    }

    constexpr int    local_size     = 128;
    constexpr size_t roles          = 2;
    constexpr size_t record_bytes   = sizeof(block_mxfp4);
    const int64_t    blocks_per_row = k / QK_MXFP4;
    const size_t     selected_count = static_cast<size_t>(n_selected);
    const size_t     rows           = static_cast<size_t>(m);
    const size_t     out_count      = rows * selected_count;
    const size_t     out_bytes      = out_count * sizeof(float);
    const size_t     weight_bytes   = kmajor_layout.size();
    const size_t     act_bytes      = pair_glu ? activations.q8_1.size() : 0;
    const size_t     total_records  = weight_bytes / record_bytes;

    uint8_t * d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t * d_act     = pair_glu ? sycl::malloc_device<uint8_t>(act_bytes, queue) : nullptr;
    float *   d_out     = sycl::malloc_device<float>(out_count, queue);

    auto cleanup = [&]() {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
    };

    if (!d_weights || !d_out || (pair_glu && !d_act)) {
        cleanup();
        error = "device allocation failed for mxfp4_selected_kmajor.";
        return false;
    }

    queue.memcpy(d_weights, kmajor_layout.data(), weight_bytes);
    if (pair_glu) {
        queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    }
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    auto submit_kernel = [&]() {
        return queue.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> partial_gate(sycl::range<1>(local_size), cgh);
            sycl::local_accessor<float, 1> partial_up(sycl::range<1>(local_size), cgh);
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(out_count, local_size), sycl::range<2>(1, local_size)),
                [=](sycl::nd_item<2> item) {
                    const size_t task = item.get_group(0);
                    const size_t sel  = task / rows;
                    const size_t row  = task - sel * rows;
                    const size_t lid  = item.get_local_id(1);

                    float gate_sum = 0.0f;
                    float up_sum   = 0.0f;
                    if (tile_read) {
                        const size_t linear_id     = task * static_cast<size_t>(local_size) + lid;
                        const size_t linear_stride = out_count * static_cast<size_t>(local_size);
                        for (size_t record_idx = linear_id; record_idx < total_records; record_idx += linear_stride) {
                            const uint8_t * record = d_weights + record_idx * record_bytes;
                            gate_sum += mxfp4_selected_kmajor_record_checksum(record);
                        }
                    } else {
                        for (int64_t block = static_cast<int64_t>(lid); block < blocks_per_row; block += local_size) {
                            const size_t base_slot =
                                ((static_cast<size_t>(block) * selected_count + sel) * roles) * rows + row;
                            const uint8_t * gate_record = d_weights + base_slot * record_bytes;
                            const uint8_t * up_record   = d_weights + (base_slot + rows) * record_bytes;

                            if (pair_glu) {
                                const uint8_t * q8_row = d_act;
                                const int8_t *  q8_qs =
                                    reinterpret_cast<const int8_t *>(q8_row) + static_cast<size_t>(block) * QK8_1;
                                const float q8_scale = mxfp4_selected_kmajor_q8_1_scale(q8_row, k, block);
                                gate_sum += mxfp4_selected_kmajor_dot_q8_1(gate_record, q8_qs, q8_scale);
                                up_sum += mxfp4_selected_kmajor_dot_q8_1(up_record, q8_qs, q8_scale);
                            } else {
                                gate_sum += mxfp4_selected_kmajor_record_checksum(gate_record);
                                gate_sum += mxfp4_selected_kmajor_record_checksum(up_record);
                            }
                        }
                    }

                    partial_gate[lid] = gate_sum;
                    partial_up[lid]   = up_sum;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (size_t stride = local_size / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            partial_gate[lid] += partial_gate[lid + stride];
                            partial_up[lid] += partial_up[lid + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (lid == 0) {
                        d_out[task] =
                            pair_glu ? mxfp4_selected_kmajor_oai_glu(partial_gate[0], partial_up[0]) : partial_gate[0];
                    }
                });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_kernel();
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit_kernel();
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops =
        pair_glu ? 4.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(k) : 0.0;
    const double bytes =
        static_cast<double>(weight_bytes) + static_cast<double>(act_bytes) + static_cast<double>(out_bytes);

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = (mean_s > 0.0 && ops > 0.0) ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();

        double max_err = 0.0;
        double sum_err = 0.0;
        for (size_t sel = 0; sel < selected_count; ++sel) {
            for (size_t row = 0; row < rows; ++row) {
                float        expected_gate = 0.0f;
                float        expected_up   = 0.0f;
                const size_t task          = sel * rows + row;
                if (tile_read) {
                    const size_t linear_stride = out_count * static_cast<size_t>(local_size);
                    for (size_t lid = 0; lid < static_cast<size_t>(local_size); ++lid) {
                        const size_t linear_id = task * static_cast<size_t>(local_size) + lid;
                        for (size_t record_idx = linear_id; record_idx < total_records; record_idx += linear_stride) {
                            const uint8_t * record = kmajor_layout.data() + record_idx * record_bytes;
                            expected_gate += mxfp4_selected_kmajor_record_checksum(record);
                        }
                    }
                } else {
                    for (int64_t block = 0; block < blocks_per_row; ++block) {
                        const size_t base_slot =
                            ((static_cast<size_t>(block) * selected_count + sel) * roles) * rows + row;
                        const uint8_t * gate_record = kmajor_layout.data() + base_slot * record_bytes;
                        const uint8_t * up_record   = kmajor_layout.data() + (base_slot + rows) * record_bytes;

                        if (pair_glu) {
                            const uint8_t * q8_row = activations.q8_1.data();
                            const int8_t *  q8_qs =
                                reinterpret_cast<const int8_t *>(q8_row) + static_cast<size_t>(block) * QK8_1;
                            const float q8_scale = mxfp4_selected_kmajor_q8_1_scale(q8_row, k, block);
                            expected_gate += mxfp4_selected_kmajor_dot_q8_1(gate_record, q8_qs, q8_scale);
                            expected_up += mxfp4_selected_kmajor_dot_q8_1(up_record, q8_qs, q8_scale);
                        } else {
                            expected_gate += mxfp4_selected_kmajor_record_checksum(gate_record);
                            expected_gate += mxfp4_selected_kmajor_record_checksum(up_record);
                        }
                    }
                }

                float expected = expected_gate;
                if (pair_glu) {
                    constexpr float alpha        = 1.702f;
                    constexpr float limit        = 7.0f;
                    const float     gate_limited = std::fmin(expected_gate, limit);
                    const float     up_limited   = std::fmax(std::fmin(expected_up, limit), -limit);
                    expected = (gate_limited / (1.0f + std::exp(-gate_limited * alpha))) * (1.0f + up_limited);
                }

                const size_t idx  = sel * rows + row;
                const double diff = std::fabs(static_cast<double>(actual[idx]) - static_cast<double>(expected));
                max_err           = std::max(max_err, diff);
                sum_err += diff;
                const double tol = pair_glu ? (1e-2 + 1e-2 * std::fabs(static_cast<double>(expected))) : 1e-3;
                if (!std::isfinite(actual[idx]) || diff > tol) {
                    cleanup();
                    char msg[256];
                    std::snprintf(msg, sizeof(msg),
                                  "mxfp4_selected_kmajor validation failed at %zu: actual=%.6f expected=%.6f "
                                  "diff=%.6f tol=%.6f max=%.6f mean=%.6f.",
                                  idx, static_cast<double>(actual[idx]), static_cast<double>(expected), diff, tol,
                                  max_err, sum_err / static_cast<double>(idx + 1));
                    error = msg;
                    return false;
                }
            }
        }
    }

    cleanup();
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
                        bool                         xmx_tiled,
                        bool                         xmx_tiled_grouped,
                        bool                         xmx_tiled_pack_q8,
                        bool                         xmx_tiled_prefetch,
                        int                          xmx_tiled_m_tiles,
                        bool                         split_gate_up,
                        bool                         single_column_gateup,
                        bool                         multi_rhs_gateup,
                        int                          multi_rhs_cols,
                        bool                         predecoded_i8,
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
    if (predecoded_i8 && !split_gate_up) {
        error = "mxfp4_pair_glu predecoded_i8 requires split gate/up.";
        return false;
    }
    if (single_column_gateup &&
        (!xmx_tiled || xmx_tiled_grouped || xmx_tiled_pack_q8 || xmx_tiled_prefetch || xmx_tiled_m_tiles != 1 ||
         (rows_per_wg != 1 && rows_per_wg != 2 && rows_per_wg != 4))) {
        error = "mxfp4_pair_glu single-column gate/up requires un-packed XMX_TILED r1/r2/r4.";
        return false;
    }
    if (multi_rhs_gateup &&
        (!xmx_tiled || xmx_tiled_grouped || xmx_tiled_pack_q8 || xmx_tiled_prefetch || xmx_tiled_m_tiles != 1 ||
         rows_per_wg != 8)) {
        error = "mxfp4_pair_glu multi-RHS gate/up requires un-packed XMX_TILED r8.";
        return false;
    }
    if (multi_rhs_gateup && multi_rhs_cols != 2 && multi_rhs_cols != 4) {
        error = "mxfp4_pair_glu multi-RHS gate/up requires n2 or n4.";
        return false;
    }
    if (xmx_tiled && (direct_xmx || split_gate_up || predecoded_i8 || vector_qs_load || scale_stride_blocks > 0)) {
        error =
            "mxfp4_pair_glu XMX_TILED path is exclusive with SOA-XMX/split/predecoded/vector/scale-stride variants.";
        return false;
    }
    if (xmx_tiled_grouped && (!xmx_tiled || xmx_tiled_pack_q8 || rows_per_wg != 8 || xmx_tiled_m_tiles != 1)) {
        error = "mxfp4_pair_glu grouped XMX_TILED path requires un-packed xmx_tiled r8.";
        return false;
    }
    if (xmx_tiled_pack_q8 && !xmx_tiled) {
        error = "mxfp4_pair_glu packed-Q8 variant requires XMX_TILED.";
        return false;
    }
    if (xmx_tiled_prefetch && (!xmx_tiled || !xmx_tiled_pack_q8)) {
        error = "mxfp4_pair_glu prefetch variant requires packed XMX_TILED.";
        return false;
    }
    if (xmx_tiled_prefetch && xmx_tiled_m_tiles != 2) {
        error = "mxfp4_pair_glu prefetch variant is only implemented for M2.";
        return false;
    }
    if (xmx_tiled_m_tiles != 1 && xmx_tiled_m_tiles != 2 && xmx_tiled_m_tiles != 4) {
        error = "mxfp4_pair_glu XMX_TILED M-tile grouping must be 1, 2, or 4.";
        return false;
    }
    if (xmx_tiled_m_tiles != 1 && (!xmx_tiled || rows_per_wg != 8)) {
        error = "mxfp4_pair_glu XMX_TILED M-tile grouping is only implemented for r8.";
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
    if (xmx_tiled && xmx_tiles_n != 1 && xmx_tiles_n != 2 && xmx_tiles_n != 4) {
        error = "mxfp4_pair_glu XMX_TILED tile count must be 1, 2, or 4.";
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
    int                  tiles_n = 0;
    if (xmx_tiled && !select_mxfp4_xmx_tiles_n(queue, xmx_tiles_n, tiles_n, error)) {
        return false;
    }
    if (predecoded_i8) {
        if (!make_mxfp4_predecoded_i8_layout(expert_layout, m, k, scale_stride, launch_layout, error)) {
            return false;
        }
    } else if (xmx_tiled) {
        if (!make_mxfp4_xmx_tiled_layout(expert_layout, m, k, tiles_n * static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N),
                                         launch_layout, error)) {
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
    const size_t  bias_count           = static_cast<size_t>(m) * expert_slots;
    const size_t  bias_bytes           = bias_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);
    const bool    validate_fused_down_q8 =
        validate && xmx_tiled && xmx_tiled_pack_q8 && xmx_tiled_m_tiles == 4 && (m % QK8_1) == 0;
    const int64_t fused_down_q8_row_bytes = static_cast<int64_t>(m * sizeof(block_q8_1) / QK8_1);
    const size_t  fused_down_q8_bytes =
        validate_fused_down_q8 ? ids_count * static_cast<size_t>(fused_down_q8_row_bytes) : 0;
    const int64_t k_tiles      = k / GGML_SYCL_MXFP4_MOE_XMX_K;
    const size_t  dpas_b_bytes = xmx_tiled_pack_q8 ?
                                     ids_count * static_cast<size_t>(k_tiles) *
                                        static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_K * GGML_SYCL_MXFP4_MOE_XMX_N) :
                                     0;
    const size_t  dpas_y_bytes = xmx_tiled_pack_q8 ? ids_count * static_cast<size_t>(k_tiles) *
                                                        static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_N) * sizeof(float) :
                                                     0;

    std::vector<uint8_t> gate_slices(weight_bytes);
    std::vector<uint8_t> up_slices(weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(launch_layout.begin(), launch_layout.end(), gate_slices.begin() + slot * expert_bytes);
        std::copy(launch_layout.begin(), launch_layout.end(), up_slices.begin() + slot * expert_bytes);
    }

    uint8_t *        d_gate            = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t *        d_up              = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_gate_ptrs       = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_up_ptrs         = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const bool       needs_ref_weights = predecoded_i8 || xmx_tiled;
    uint8_t *        d_ref_gate =
        needs_ref_weights ? sycl::malloc_device<uint8_t>(expert_layout.size() * expert_slots, queue) : nullptr;
    uint8_t * d_ref_up =
        needs_ref_weights ? sycl::malloc_device<uint8_t>(expert_layout.size() * expert_slots, queue) : nullptr;
    const uint8_t ** d_ref_gate_ptrs =
        needs_ref_weights ? sycl::malloc_device<const uint8_t *>(expert_slots, queue) : nullptr;
    const uint8_t ** d_ref_up_ptrs =
        needs_ref_weights ? sycl::malloc_device<const uint8_t *>(expert_slots, queue) : nullptr;
    uint8_t * d_act                = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t * d_ids                = sycl::malloc_device<int32_t>(ids_count, queue);
    int32_t * d_grouped_experts    = nullptr;
    int32_t * d_grouped_offsets    = nullptr;
    int32_t * d_grouped_rows       = nullptr;
    int32_t * d_grouped_chunks     = nullptr;
    int32_t * d_grouped_row_starts = nullptr;
    float *   d_out                = sycl::malloc_device<float>(out_count, queue);
    float *   d_gate_bias          = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    float *   d_up_bias            = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    uint8_t * d_fused_down_q8 =
        validate_fused_down_q8 ? sycl::malloc_device<uint8_t>(fused_down_q8_bytes, queue) : nullptr;
    const bool need_ref_out =
        validate && (direct_xmx || xmx_tiled || xmx_tiled_grouped || split_gate_up || vector_qs_load);
    float *  d_ref_out  = need_ref_out ? sycl::malloc_device<float>(out_count, queue) : nullptr;
    float *  d_gate_tmp = split_gate_up ? sycl::malloc_device<float>(out_count, queue) : nullptr;
    float *  d_up_tmp   = split_gate_up ? sycl::malloc_device<float>(out_count, queue) : nullptr;
    int8_t * d_dpas_b =
        xmx_tiled_pack_q8 ? sycl::malloc_device<int8_t>(dpas_b_bytes == 0 ? 1 : dpas_b_bytes, queue) : nullptr;
    float * d_dpas_y =
        xmx_tiled_pack_q8 ?
            sycl::malloc_device<float>((dpas_y_bytes == 0 ? sizeof(float) : dpas_y_bytes) / sizeof(float), queue) :
            nullptr;

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
        if (d_ref_gate) {
            sycl::free(d_ref_gate, queue);
        }
        if (d_ref_up) {
            sycl::free(d_ref_up, queue);
        }
        if (d_ref_gate_ptrs) {
            sycl::free(d_ref_gate_ptrs, queue);
        }
        if (d_ref_up_ptrs) {
            sycl::free(d_ref_up_ptrs, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_ids) {
            sycl::free(d_ids, queue);
        }
        if (d_grouped_experts) {
            sycl::free(d_grouped_experts, queue);
        }
        if (d_grouped_offsets) {
            sycl::free(d_grouped_offsets, queue);
        }
        if (d_grouped_rows) {
            sycl::free(d_grouped_rows, queue);
        }
        if (d_grouped_chunks) {
            sycl::free(d_grouped_chunks, queue);
        }
        if (d_grouped_row_starts) {
            sycl::free(d_grouped_row_starts, queue);
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
        if (d_fused_down_q8) {
            sycl::free(d_fused_down_q8, queue);
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
        if (d_dpas_b) {
            sycl::free(d_dpas_b, queue);
        }
        if (d_dpas_y) {
            sycl::free(d_dpas_y, queue);
        }
    };

    if (!d_gate || !d_up || !d_gate_ptrs || !d_up_ptrs || !d_act || !d_ids || !d_out ||
        (use_bias && (!d_gate_bias || !d_up_bias)) || (validate_fused_down_q8 && !d_fused_down_q8) ||
        (need_ref_out && !d_ref_out) || (split_gate_up && (!d_gate_tmp || !d_up_tmp)) ||
        (xmx_tiled_pack_q8 && (!d_dpas_b || !d_dpas_y)) ||
        (needs_ref_weights && (!d_ref_gate || !d_ref_up || !d_ref_gate_ptrs || !d_ref_up_ptrs))) {
        cleanup();
        error = "device allocation failed for mxfp4_pair_glu.";
        return false;
    }

    std::vector<const uint8_t *> host_gate_ptrs(expert_slots);
    std::vector<const uint8_t *> host_up_ptrs(expert_slots);
    std::vector<const uint8_t *> host_ref_gate_ptrs(expert_slots);
    std::vector<const uint8_t *> host_ref_up_ptrs(expert_slots);
    std::vector<int32_t>         host_ids(ids_count);
    std::vector<float>           host_gate_bias;
    std::vector<float>           host_up_bias;
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_gate_ptrs[slot] = d_gate + slot * expert_bytes;
        host_up_ptrs[slot]   = d_up + slot * expert_bytes;
        if (needs_ref_weights) {
            host_ref_gate_ptrs[slot] = d_ref_gate + slot * expert_layout.size();
            host_ref_up_ptrs[slot]   = d_ref_up + slot * expert_layout.size();
        }
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            size_t slot;
            if (multi_rhs_gateup) {
                const size_t group_base_sel = (sel / static_cast<size_t>(multi_rhs_cols)) *
                                              static_cast<size_t>(multi_rhs_cols);
                slot = sparse_expert_slots ? sparse_expert_slot(group_base_sel, selected_count, expert_slots) :
                                             group_base_sel;
            } else {
                slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            }
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    std::vector<int32_t> grouped_experts_host;
    std::vector<int32_t> grouped_offsets_host;
    std::vector<int32_t> grouped_rows_host;
    std::vector<int32_t> grouped_chunks_host;
    std::vector<int32_t> grouped_row_starts_host;
    if (xmx_tiled_grouped) {
        constexpr int                     exec_n         = GGML_SYCL_MXFP4_MOE_XMX_N;
        constexpr int                     row_list_tiles = 16;
        const size_t                      row_limit      = static_cast<size_t>(exec_n * row_list_tiles);
        std::vector<std::vector<int32_t>> grouped_slots;
        grouped_slots.reserve(std::min(expert_slots, ids_count));
        for (size_t token = 0; token < token_count; ++token) {
            for (size_t sel = 0; sel < selected_count; ++sel) {
                const int32_t eid = host_ids[token * selected_count + sel];
                auto          it  = std::find(grouped_experts_host.begin(), grouped_experts_host.end(), eid);
                size_t        group_index;
                if (it == grouped_experts_host.end()) {
                    group_index = grouped_experts_host.size();
                    grouped_experts_host.push_back(eid);
                    grouped_slots.emplace_back();
                } else {
                    group_index = static_cast<size_t>(std::distance(grouped_experts_host.begin(), it));
                }
                grouped_slots[group_index].push_back(static_cast<int32_t>(sel * token_count + token));
            }
        }
        grouped_offsets_host.reserve(grouped_experts_host.size() + 1);
        grouped_offsets_host.push_back(0);
        for (size_t group = 0; group < grouped_slots.size(); ++group) {
            auto & rows = grouped_slots[group];
            std::sort(rows.begin(), rows.end());
            const int rows_begin = static_cast<int>(grouped_rows_host.size());
            grouped_rows_host.insert(grouped_rows_host.end(), rows.begin(), rows.end());
            for (int row_start = 0; row_start < static_cast<int>(rows.size()); row_start += exec_n) {
                grouped_chunks_host.push_back(static_cast<int32_t>(group));
                grouped_row_starts_host.push_back(row_start);
            }
            grouped_offsets_host.push_back(rows_begin + static_cast<int>(rows.size()));
        }
        if (grouped_rows_host.size() != ids_count || grouped_chunks_host.empty() || row_limit == 0) {
            cleanup();
            error = "mxfp4_pair_glu grouped XMX_TILED setup failed.";
            return false;
        }
        d_grouped_experts    = sycl::malloc_device<int32_t>(grouped_experts_host.size(), queue);
        d_grouped_offsets    = sycl::malloc_device<int32_t>(grouped_offsets_host.size(), queue);
        d_grouped_rows       = sycl::malloc_device<int32_t>(grouped_rows_host.size(), queue);
        d_grouped_chunks     = sycl::malloc_device<int32_t>(grouped_chunks_host.size(), queue);
        d_grouped_row_starts = sycl::malloc_device<int32_t>(grouped_row_starts_host.size(), queue);
        if (!d_grouped_experts || !d_grouped_offsets || !d_grouped_rows || !d_grouped_chunks || !d_grouped_row_starts) {
            cleanup();
            error = "device allocation failed for grouped mxfp4_pair_glu.";
            return false;
        }
        queue.memcpy(d_grouped_experts, grouped_experts_host.data(), grouped_experts_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_offsets, grouped_offsets_host.data(), grouped_offsets_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_rows, grouped_rows_host.data(), grouped_rows_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_chunks, grouped_chunks_host.data(), grouped_chunks_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_row_starts, grouped_row_starts_host.data(),
                     grouped_row_starts_host.size() * sizeof(int32_t));
    }

    queue.memcpy(d_gate, gate_slices.data(), weight_bytes);
    queue.memcpy(d_up, up_slices.data(), weight_bytes);
    queue.memcpy(d_gate_ptrs, host_gate_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_up_ptrs, host_up_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    if (needs_ref_weights) {
        std::vector<uint8_t> ref_gate_slices(expert_layout.size() * expert_slots);
        std::vector<uint8_t> ref_up_slices(expert_layout.size() * expert_slots);
        for (size_t slot = 0; slot < expert_slots; ++slot) {
            std::copy(expert_layout.begin(), expert_layout.end(),
                      ref_gate_slices.begin() + slot * expert_layout.size());
            std::copy(expert_layout.begin(), expert_layout.end(), ref_up_slices.begin() + slot * expert_layout.size());
        }
        queue.memcpy(d_ref_gate, ref_gate_slices.data(), ref_gate_slices.size());
        queue.memcpy(d_ref_up, ref_up_slices.data(), ref_up_slices.size());
        queue.memcpy(d_ref_gate_ptrs, host_ref_gate_ptrs.data(), expert_slots * sizeof(const uint8_t *));
        queue.memcpy(d_ref_up_ptrs, host_ref_up_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    }
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
    if (validate_fused_down_q8) {
        queue.memset(d_fused_down_q8, 0, fused_down_q8_bytes);
    }
    queue.wait_and_throw();

    ggml_sycl::mxfp4_pair_glu_bench_args args{};
    args.stream              = &queue;
    args.gate_ptrs           = reinterpret_cast<const void * const *>(d_gate_ptrs);
    args.up_ptrs             = reinterpret_cast<const void * const *>(d_up_ptrs);
    args.activations_q8_soa  = d_act;
    args.output              = d_out;
    args.gate_tmp            = d_gate_tmp;
    args.up_tmp              = d_up_tmp;
    args.down_q8_soa         = validate_fused_down_q8 ? d_fused_down_q8 : nullptr;
    args.dpas_b_packed       = d_dpas_b;
    args.dpas_y_scales       = d_dpas_y;
    args.ids                 = d_ids;
    args.grouped_expert_ids  = d_grouped_experts;
    args.grouped_offsets     = d_grouped_offsets;
    args.grouped_row_slots   = d_grouped_rows;
    args.grouped_chunks      = d_grouped_chunks;
    args.grouped_row_starts  = d_grouped_row_starts;
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
    args.down_q8_nb11        = validate_fused_down_q8 ? fused_down_q8_row_bytes : 0;
    args.gate_bias           = d_gate_bias;
    args.up_bias             = d_up_bias;
    args.gate_bias_nb1       = static_cast<int64_t>(m * sizeof(float));
    args.up_bias_nb1         = static_cast<int64_t>(m * sizeof(float));
    args.rows_per_wg         = rows_per_wg;
    args.cache_y             = cache_y;
    args.direct_xmx          = direct_xmx;
    args.xmx_tiled           = xmx_tiled;
    args.xmx_tiled_grouped   = xmx_tiled_grouped;
    args.xmx_tiled_pack_q8   = xmx_tiled_pack_q8;
    args.xmx_tiled_prefetch   = xmx_tiled_prefetch;
    args.xmx_tiled_m_tiles    = xmx_tiled_m_tiles;
    args.split_gate_up        = split_gate_up;
    args.single_column_gateup = single_column_gateup;
    args.multi_rhs_gateup     = multi_rhs_gateup;
    args.multi_rhs_cols       = multi_rhs_cols;
    args.predecoded_i8        = predecoded_i8;
    args.xmx_tiles_n          = xmx_tiled ? tiles_n : xmx_tiles_n;
    args.vector_qs_load      = vector_qs_load;
    args.ignore_weight_scale = ignore_weight_scale;
    args.scale_stride_blocks = scale_stride_blocks;
    args.subgroup_size       = subgroup_size;
    args.grouped_n_chunks    = static_cast<int>(grouped_chunks_host.size());
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
        static_cast<double>(fused_down_q8_bytes) + (split_gate_up ? 4.0 * static_cast<double>(out_bytes) : 0.0) +
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
        if (validate_fused_down_q8) {
            std::vector<uint8_t> fused_down_q8(fused_down_q8_bytes);
            queue.memcpy(fused_down_q8.data(), d_fused_down_q8, fused_down_q8_bytes).wait();
            if (!validate_q8_1_soa_from_f32_rows(actual, fused_down_q8, static_cast<int64_t>(ids_count), m,
                                                 fused_down_q8_row_bytes, error)) {
                cleanup();
                return false;
            }
        }
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
            if (needs_ref_weights) {
                ref_args.gate_ptrs = reinterpret_cast<const void * const *>(d_ref_gate_ptrs);
                ref_args.up_ptrs   = reinterpret_cast<const void * const *>(d_ref_up_ptrs);
            }
            ref_args.direct_xmx         = false;
            ref_args.xmx_tiled          = false;
            ref_args.xmx_tiled_grouped  = false;
            ref_args.xmx_tiled_pack_q8  = false;
            ref_args.xmx_tiled_prefetch = false;
            ref_args.xmx_tiled_m_tiles  = 1;
            ref_args.split_gate_up       = false;
            ref_args.single_column_gateup = false;
            ref_args.multi_rhs_gateup    = false;
            ref_args.multi_rhs_cols      = 1;
            ref_args.predecoded_i8       = false;
            ref_args.gate_tmp           = nullptr;
            ref_args.up_tmp             = nullptr;
            ref_args.down_q8_soa        = nullptr;
            ref_args.down_q8_nb11       = 0;
            ref_args.dpas_b_packed      = nullptr;
            ref_args.dpas_y_scales      = nullptr;
            ref_args.grouped_expert_ids = nullptr;
            ref_args.grouped_offsets    = nullptr;
            ref_args.grouped_row_slots  = nullptr;
            ref_args.grouped_chunks     = nullptr;
            ref_args.grouped_row_starts = nullptr;
            ref_args.grouped_n_chunks   = 0;
            ref_args.xmx_tiles_n        = 1;
            ref_args.vector_qs_load     = false;
            ref_args.rows_per_wg        = 1;
            ref_args.cache_y            = false;
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
                              bool                         xmx_tiled_gate_up,
                              bool                         xmx_tiled_grouped,
                              bool                         xmx_tiled_pack_q8,
                              bool                         xmx_tiled_prefetch,
                              int                          xmx_tiled_m_tiles,
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
    if (xmx_tiled_grouped && (!xmx_tiled_gate_up || rows_per_wg != 8)) {
        error = "mxfp4_layer_glu_down grouped XMX_TILED gate/up requires xmx_tiled r8.";
        return false;
    }
    if (xmx_tiled_pack_q8 && (!xmx_tiled_gate_up || xmx_tiled_grouped)) {
        error = "mxfp4_layer_glu_down packed-Q8 variant requires non-grouped XMX_TILED.";
        return false;
    }
    if (xmx_tiled_prefetch && (!xmx_tiled_gate_up || !xmx_tiled_pack_q8)) {
        error = "mxfp4_layer_glu_down prefetch variant requires packed XMX_TILED.";
        return false;
    }
    if (xmx_tiled_prefetch && xmx_tiled_m_tiles != 2) {
        error = "mxfp4_layer_glu_down prefetch variant is only implemented for M2.";
        return false;
    }
    if (xmx_tiled_m_tiles != 1 && xmx_tiled_m_tiles != 2 && xmx_tiled_m_tiles != 4) {
        error = "mxfp4_layer_glu_down XMX_TILED M-tile grouping must be 1, 2, or 4.";
        return false;
    }
    if (xmx_tiled_m_tiles != 1 && (!xmx_tiled_gate_up || rows_per_wg != 8)) {
        error = "mxfp4_layer_glu_down XMX_TILED M-tile grouping is only implemented for r8.";
        return false;
    }
    if (xmx_tiled_gate_up && (vector_qs_load || scale_stride_blocks > 0)) {
        error = "mxfp4_layer_glu_down XMX_TILED gate/up is exclusive with SOA load variants.";
        return false;
    }
    if (xmx_tiled_gate_up && xmx_tiles_n != 1 && xmx_tiles_n != 2 && xmx_tiles_n != 4) {
        error = "mxfp4_layer_glu_down XMX_TILED tile count must be 1, 2, or 4.";
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
    std::vector<uint8_t> gate_up_layout;
    int                  gate_up_tiles_n = 0;
    if (xmx_tiled_gate_up && !select_mxfp4_xmx_tiles_n(queue, xmx_tiles_n, gate_up_tiles_n, error)) {
        return false;
    }
    if (xmx_tiled_gate_up) {
        if (!make_mxfp4_xmx_tiled_layout(expert_layout, m, k,
                                         gate_up_tiles_n * static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N), gate_up_layout,
                                         error)) {
            return false;
        }
    } else {
        gate_up_layout = expert_layout;
    }
    const size_t  gate_up_expert_bytes    = gate_up_layout.size();
    const size_t  down_expert_bytes       = expert_layout.size();
    const size_t  logical_expert_bytes    = weights.layout.size();
    const size_t  expert_slots            = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  gate_up_weight_bytes    = gate_up_expert_bytes * expert_slots;
    const size_t  down_weight_bytes       = down_expert_bytes * expert_slots;
    const size_t  act_bytes               = activations.q8_1.size();
    const size_t  glu_count               = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  glu_bytes               = glu_count * sizeof(float);
    const size_t  out_count               = glu_count;
    const size_t  out_bytes               = out_count * sizeof(float);
    const size_t  bias_count              = static_cast<size_t>(m) * expert_slots;
    const size_t  bias_bytes              = bias_count * sizeof(float);
    const int64_t activation_q8_row_bytes = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);
    const int64_t down_q8_row_bytes       = static_cast<int64_t>(m * sizeof(block_q8_1) / QK8_1);
    const size_t  down_q8_bytes           = ids_count * static_cast<size_t>(down_q8_row_bytes);
    const int64_t k_tiles                 = k / GGML_SYCL_MXFP4_MOE_XMX_K;
    const size_t  dpas_b_bytes            = xmx_tiled_pack_q8 ?
                                                ids_count * static_cast<size_t>(k_tiles) *
                                        static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_K * GGML_SYCL_MXFP4_MOE_XMX_N) :
                                                0;
    const size_t  dpas_y_bytes            = xmx_tiled_pack_q8 ? ids_count * static_cast<size_t>(k_tiles) *
                                                        static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_N) * sizeof(float) :
                                                                0;

    std::vector<uint8_t> gate_slices(gate_up_weight_bytes);
    std::vector<uint8_t> up_slices(gate_up_weight_bytes);
    std::vector<uint8_t> down_slices(down_weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(gate_up_layout.begin(), gate_up_layout.end(), gate_slices.begin() + slot * gate_up_expert_bytes);
        std::copy(gate_up_layout.begin(), gate_up_layout.end(), up_slices.begin() + slot * gate_up_expert_bytes);
        std::copy(expert_layout.begin(), expert_layout.end(), down_slices.begin() + slot * down_expert_bytes);
    }

    uint8_t *        d_gate               = sycl::malloc_device<uint8_t>(gate_up_weight_bytes, queue);
    uint8_t *        d_up                 = sycl::malloc_device<uint8_t>(gate_up_weight_bytes, queue);
    uint8_t *        d_down               = sycl::malloc_device<uint8_t>(down_weight_bytes, queue);
    const uint8_t ** d_gate_ptrs          = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_up_ptrs            = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    const uint8_t ** d_down_ptrs          = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_act                = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids                = sycl::malloc_device<int32_t>(ids_count, queue);
    int32_t *        d_grouped_experts    = nullptr;
    int32_t *        d_grouped_offsets    = nullptr;
    int32_t *        d_grouped_rows       = nullptr;
    int32_t *        d_grouped_chunks     = nullptr;
    int32_t *        d_grouped_row_starts = nullptr;
    float *          d_glu                = sycl::malloc_device<float>(glu_count, queue);
    uint8_t *        d_down_q8            = sycl::malloc_device<uint8_t>(down_q8_bytes, queue);
    float *          d_out                = sycl::malloc_device<float>(out_count, queue);
    float *          d_gate_bias          = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    float *          d_up_bias            = use_bias ? sycl::malloc_device<float>(bias_count, queue) : nullptr;
    int8_t *         d_dpas_b =
        xmx_tiled_pack_q8 ? sycl::malloc_device<int8_t>(dpas_b_bytes == 0 ? 1 : dpas_b_bytes, queue) : nullptr;
    float * d_dpas_y =
        xmx_tiled_pack_q8 ?
            sycl::malloc_device<float>((dpas_y_bytes == 0 ? sizeof(float) : dpas_y_bytes) / sizeof(float), queue) :
            nullptr;

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
        if (d_grouped_experts) {
            sycl::free(d_grouped_experts, queue);
        }
        if (d_grouped_offsets) {
            sycl::free(d_grouped_offsets, queue);
        }
        if (d_grouped_rows) {
            sycl::free(d_grouped_rows, queue);
        }
        if (d_grouped_chunks) {
            sycl::free(d_grouped_chunks, queue);
        }
        if (d_grouped_row_starts) {
            sycl::free(d_grouped_row_starts, queue);
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
        if (d_dpas_b) {
            sycl::free(d_dpas_b, queue);
        }
        if (d_dpas_y) {
            sycl::free(d_dpas_y, queue);
        }
    };

    if (!d_gate || !d_up || !d_down || !d_gate_ptrs || !d_up_ptrs || !d_down_ptrs || !d_act || !d_ids || !d_glu ||
        !d_down_q8 || !d_out || (use_bias && (!d_gate_bias || !d_up_bias)) ||
        (xmx_tiled_pack_q8 && (!d_dpas_b || !d_dpas_y))) {
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
        host_gate_ptrs[slot] = d_gate + slot * gate_up_expert_bytes;
        host_up_ptrs[slot]   = d_up + slot * gate_up_expert_bytes;
        host_down_ptrs[slot] = d_down + slot * down_expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    std::vector<int32_t> grouped_experts_host;
    std::vector<int32_t> grouped_offsets_host;
    std::vector<int32_t> grouped_rows_host;
    std::vector<int32_t> grouped_chunks_host;
    std::vector<int32_t> grouped_row_starts_host;
    if (xmx_tiled_grouped) {
        constexpr int                     exec_n         = GGML_SYCL_MXFP4_MOE_XMX_N;
        constexpr int                     row_list_tiles = 16;
        const size_t                      row_limit      = static_cast<size_t>(exec_n * row_list_tiles);
        std::vector<std::vector<int32_t>> grouped_slots;
        grouped_slots.reserve(std::min(expert_slots, ids_count));
        for (size_t token = 0; token < token_count; ++token) {
            for (size_t sel = 0; sel < selected_count; ++sel) {
                const int32_t eid = host_ids[token * selected_count + sel];
                auto          it  = std::find(grouped_experts_host.begin(), grouped_experts_host.end(), eid);
                size_t        group_index;
                if (it == grouped_experts_host.end()) {
                    group_index = grouped_experts_host.size();
                    grouped_experts_host.push_back(eid);
                    grouped_slots.emplace_back();
                } else {
                    group_index = static_cast<size_t>(std::distance(grouped_experts_host.begin(), it));
                }
                grouped_slots[group_index].push_back(static_cast<int32_t>(sel * token_count + token));
            }
        }
        grouped_offsets_host.reserve(grouped_experts_host.size() + 1);
        grouped_offsets_host.push_back(0);
        for (size_t group = 0; group < grouped_slots.size(); ++group) {
            auto & rows = grouped_slots[group];
            std::sort(rows.begin(), rows.end());
            const int rows_begin = static_cast<int>(grouped_rows_host.size());
            grouped_rows_host.insert(grouped_rows_host.end(), rows.begin(), rows.end());
            for (int row_start = 0; row_start < static_cast<int>(rows.size()); row_start += exec_n) {
                grouped_chunks_host.push_back(static_cast<int32_t>(group));
                grouped_row_starts_host.push_back(row_start);
            }
            grouped_offsets_host.push_back(rows_begin + static_cast<int>(rows.size()));
        }
        if (grouped_rows_host.size() != ids_count || grouped_chunks_host.empty() || row_limit == 0) {
            cleanup();
            error = "mxfp4_layer_glu_down grouped XMX_TILED setup failed.";
            return false;
        }
        d_grouped_experts    = sycl::malloc_device<int32_t>(grouped_experts_host.size(), queue);
        d_grouped_offsets    = sycl::malloc_device<int32_t>(grouped_offsets_host.size(), queue);
        d_grouped_rows       = sycl::malloc_device<int32_t>(grouped_rows_host.size(), queue);
        d_grouped_chunks     = sycl::malloc_device<int32_t>(grouped_chunks_host.size(), queue);
        d_grouped_row_starts = sycl::malloc_device<int32_t>(grouped_row_starts_host.size(), queue);
        if (!d_grouped_experts || !d_grouped_offsets || !d_grouped_rows || !d_grouped_chunks || !d_grouped_row_starts) {
            cleanup();
            error = "device allocation failed for grouped mxfp4_layer_glu_down.";
            return false;
        }
        queue.memcpy(d_grouped_experts, grouped_experts_host.data(), grouped_experts_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_offsets, grouped_offsets_host.data(), grouped_offsets_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_rows, grouped_rows_host.data(), grouped_rows_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_chunks, grouped_chunks_host.data(), grouped_chunks_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_row_starts, grouped_row_starts_host.data(),
                     grouped_row_starts_host.size() * sizeof(int32_t));
    }

    queue.memcpy(d_gate, gate_slices.data(), gate_up_weight_bytes);
    queue.memcpy(d_up, up_slices.data(), gate_up_weight_bytes);
    queue.memcpy(d_down, down_slices.data(), down_weight_bytes);
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
    args.activation_nb11        = activation_q8_row_bytes;
    args.activation_nb12        = activation_q8_row_bytes;
    args.glu_nb1                = static_cast<int64_t>(m * sizeof(float));
    args.glu_nb2                = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.down_q8_nb11           = down_q8_row_bytes;
    args.down_q8_nb12           = static_cast<int64_t>(selected_count) * down_q8_row_bytes;
    args.output_nb1             = static_cast<int64_t>(m * sizeof(float));
    args.output_nb2             = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.gate_bias_nb1          = static_cast<int64_t>(m * sizeof(float));
    args.up_bias_nb1            = static_cast<int64_t>(m * sizeof(float));
    args.rows_per_wg            = rows_per_wg;
    args.cache_y                = cache_y;
    args.xmx_tiled_gate_up      = xmx_tiled_gate_up;
    args.xmx_tiled_grouped      = xmx_tiled_grouped;
    args.xmx_tiled_pack_q8      = xmx_tiled_pack_q8;
    args.xmx_tiled_prefetch     = xmx_tiled_prefetch;
    args.xmx_tiled_m_tiles      = xmx_tiled_m_tiles;
    args.xmx_tiles_n            = xmx_tiled_gate_up ? gate_up_tiles_n : xmx_tiles_n;
    args.vector_qs_load         = vector_qs_load;
    args.ignore_weight_scale    = ignore_weight_scale;
    args.scale_stride_blocks    = scale_stride_blocks;
    args.subgroup_size          = subgroup_size;
    args.grouped_expert_ids     = d_grouped_experts;
    args.grouped_offsets        = d_grouped_offsets;
    args.grouped_row_slots      = d_grouped_rows;
    args.grouped_chunks         = d_grouped_chunks;
    args.grouped_row_starts     = d_grouped_row_starts;
    args.grouped_n_chunks       = static_cast<int>(grouped_chunks_host.size());
    args.dpas_b_packed          = d_dpas_b;
    args.dpas_y_scales          = d_dpas_y;
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
        std::vector<float>   glu(glu_count);
        std::vector<uint8_t> down_q8(down_q8_bytes);
        std::vector<float>   actual(out_count);
        queue.memcpy(glu.data(), d_glu, glu_bytes).wait();
        queue.memcpy(down_q8.data(), d_down_q8, down_q8_bytes).wait();
        queue.memcpy(actual.data(), d_out, out_bytes).wait();
        if (!validate_q8_1_soa_from_f32_rows(glu, down_q8, static_cast<int64_t>(ids_count), m, down_q8_row_bytes,
                                             error)) {
            cleanup();
            return false;
        }
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

template <int LocalSize> struct mxfp4_mmv_id_f32_kernel;

template <int LocalSize>
static sycl::event launch_mxfp4_mmv_id_f32_kernel(const uint8_t * const * expert_ptrs,
                                                  const float *           activations_f32,
                                                  float *                 output,
                                                  const int32_t *         ids,
                                                  int                     ncols,
                                                  int                     nrows_per_expert,
                                                  int                     total_batches,
                                                  int                     n_ids,
                                                  int                     n_tokens,
                                                  int64_t                 ids_nb0,
                                                  int64_t                 ids_nb1,
                                                  int64_t                 act_nb1,
                                                  int64_t                 act_nb2,
                                                  int64_t                 dst_nb1,
                                                  int64_t                 dst_nb2,
                                                  int                     scale_stride_blocks,
                                                  bool                    apply_weight_scale,
                                                  sycl::queue &           queue) {
    constexpr int qbytes_per_block = QK_MXFP4 / 2;
    const int     blocks_per_row   = ncols / QK_MXFP4;
    const int     scale_stride     = scale_stride_blocks > 0 ? scale_stride_blocks : blocks_per_row;
    const size_t  qs_bytes         = static_cast<size_t>(nrows_per_expert) * static_cast<size_t>(blocks_per_row) *
                            static_cast<size_t>(qbytes_per_block);

    return queue.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(LocalSize), cgh);
        cgh.parallel_for<mxfp4_mmv_id_f32_kernel<LocalSize>>(
            sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(total_batches) * static_cast<size_t>(nrows_per_expert),
                                             static_cast<size_t>(LocalSize)),
                              sycl::range<2>(1, static_cast<size_t>(LocalSize))),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(LocalSize)]] {
                const size_t task  = item.get_group(0);
                const int    row   = static_cast<int>(task % static_cast<size_t>(nrows_per_expert));
                const int    batch = static_cast<int>(task / static_cast<size_t>(nrows_per_expert));
                const int    token = batch / n_ids;
                const int    sel   = batch - token * n_ids;
                const int    lid   = static_cast<int>(item.get_local_id(1));

                const auto * ids_base = reinterpret_cast<const int32_t *>(reinterpret_cast<const uint8_t *>(ids) +
                                                                          static_cast<size_t>(token) * ids_nb1);
                const int    expert   = *reinterpret_cast<const int32_t *>(reinterpret_cast<const uint8_t *>(ids_base) +
                                                                           static_cast<size_t>(sel) * ids_nb0);
                const uint8_t * weight = expert_ptrs[expert];
                const float * act = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(activations_f32) +
                                                                    static_cast<size_t>(token) * act_nb2 +
                                                                    static_cast<size_t>(sel) * act_nb1);
                float *       dst = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(output) +
                                                              static_cast<size_t>(token) * dst_nb2 +
                                                              static_cast<size_t>(sel) * dst_nb1);

                const uint8_t * qs_base     = weight;
                const uint8_t * scales_base = weight + qs_bytes;
                float           sum         = 0.0f;
                for (int block = lid; block < blocks_per_row; block += LocalSize) {
                    const size_t block_idx =
                        static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(block);
                    const uint8_t * qs = qs_base + block_idx * qbytes_per_block;
                    const float     scale =
                        apply_weight_scale ?
                                mxfp4_e8m0_to_fp32_device(
                                scales_base[static_cast<size_t>(row) * static_cast<size_t>(scale_stride) +
                                            static_cast<size_t>(block)]) :
                                1.0f;
                    const int act_base = block * QK_MXFP4;
                    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                        const uint8_t packed = qs[i];
                        sum += scale * mxfp4_value_device(packed & 0x0f) * act[act_base + i];
                        sum += scale * mxfp4_value_device(packed >> 4) * act[act_base + QK_MXFP4 / 2 + i];
                    }
                }

                partial[lid] = sum;
                item.barrier(sycl::access::fence_space::local_space);
                for (int stride = LocalSize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        partial[lid] += partial[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (lid == 0) {
                    dst[row] = partial[0];
                }
            });
    });
}

bool run_mxfp4_mmv_id_f32(const GeneratedWeights &     weights,
                          const GeneratedActivations & activations,
                          int64_t                      m,
                          int64_t                      n_selected,
                          int64_t                      k,
                          int64_t                      n_tokens,
                          bool                         ignore_weight_scale,
                          int                          scale_stride_blocks,
                          int                          subgroup_size,
                          bool                         sparse_expert_slots,
                          bool                         validate,
                          int                          warmup,
                          int                          iterations,
                          sycl::queue &                queue,
                          ReferenceMetrics &           out,
                          std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_tokens <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "mxfp4_mmv_id_f32 requires positive M/selected/tokens/K and K divisible by QK_MXFP4.";
        return false;
    }
    if (subgroup_size != 16 && subgroup_size != 32) {
        error = "mxfp4_mmv_id_f32 subgroup_size must be 16 or 32.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_mmv_id_f32 requires SOA MXFP4 weights.";
        return false;
    }
    const size_t selected_count = static_cast<size_t>(n_selected);
    const size_t token_count    = static_cast<size_t>(n_tokens);
    if (activations.fp32.size() < selected_count * token_count * static_cast<size_t>(k)) {
        error = "mxfp4_mmv_id_f32 requires F32 activations for every selected token row.";
        return false;
    }

    const int blocks_per_row = static_cast<int>(k / QK_MXFP4);
    const int scale_stride   = scale_stride_blocks > 0 ? scale_stride_blocks : blocks_per_row;
    if (scale_stride < blocks_per_row) {
        error = "mxfp4_mmv_id_f32 scale stride must be at least K / QK_MXFP4.";
        return false;
    }

    std::vector<uint8_t> expert_layout;
    if (!make_mxfp4_soa_scale_stride_layout(weights.layout, m, k, scale_stride, expert_layout, error)) {
        return false;
    }

    const size_t ids_count    = selected_count * token_count;
    const size_t expert_slots = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t weight_bytes = expert_layout.size() * expert_slots;
    const size_t act_count    = selected_count * token_count * static_cast<size_t>(k);
    const size_t act_bytes    = act_count * sizeof(float);
    const size_t out_count    = static_cast<size_t>(m) * selected_count * token_count;
    const size_t out_bytes    = out_count * sizeof(float);

    std::vector<uint8_t> weight_slices(weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(expert_layout.begin(), expert_layout.end(), weight_slices.begin() + slot * expert_layout.size());
    }

    uint8_t *        d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_ptrs    = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    float *          d_act     = sycl::malloc_device<float>(act_count, queue);
    int32_t *        d_ids     = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out     = sycl::malloc_device<float>(out_count, queue);

    auto cleanup = [&]() {
        if (d_weights) {
            sycl::free(d_weights, queue);
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
    };

    if (!d_weights || !d_ptrs || !d_act || !d_ids || !d_out) {
        cleanup();
        error = "device allocation failed for mxfp4_mmv_id_f32.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(expert_slots);
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_ptrs[slot] = d_weights + slot * expert_layout.size();
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    queue.memcpy(d_weights, weight_slices.data(), weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.fp32.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    const int     total_batches = static_cast<int>(ids_count);
    const int64_t act_nb1       = static_cast<int64_t>(k * sizeof(float));
    const int64_t act_nb2       = static_cast<int64_t>(selected_count * static_cast<size_t>(k) * sizeof(float));
    const int64_t dst_nb1       = static_cast<int64_t>(m * sizeof(float));
    const int64_t dst_nb2       = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    auto          launch        = [&]() {
        if (subgroup_size == 16) {
            return launch_mxfp4_mmv_id_f32_kernel<16>(
                d_ptrs, d_act, d_out, d_ids, static_cast<int>(k), static_cast<int>(m), total_batches,
                static_cast<int>(n_selected), static_cast<int>(n_tokens), sizeof(int32_t),
                static_cast<int64_t>(selected_count * sizeof(int32_t)), act_nb1, act_nb2, dst_nb1, dst_nb2,
                scale_stride, !ignore_weight_scale, queue);
        }
        return launch_mxfp4_mmv_id_f32_kernel<32>(d_ptrs, d_act, d_out, d_ids, static_cast<int>(k), static_cast<int>(m),
                                                                  total_batches, static_cast<int>(n_selected),
                                                                  static_cast<int>(n_tokens), sizeof(int32_t),
                                                                  static_cast<int64_t>(selected_count * sizeof(int32_t)), act_nb1,
                                                                  act_nb2, dst_nb1, dst_nb2, scale_stride, !ignore_weight_scale, queue);
    };

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us  = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s   = mean_us * 1e-6;
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) *
                       static_cast<double>(k);
    const double bytes =
        static_cast<double>(weights.layout.size()) * static_cast<double>(n_selected) * static_cast<double>(n_tokens) +
        static_cast<double>(act_bytes) + static_cast<double>(out_bytes) +
        static_cast<double>(ids_count * sizeof(int32_t)) + static_cast<double>(expert_slots * sizeof(const uint8_t *));

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
                error = "mxfp4_mmv_id_f32 validation failed: non-finite output.";
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

    const bool       needs_ref = validate && (vector_qs_load || predecoded_i8);
    uint8_t *        d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_ptrs    = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_ref_weights =
        predecoded_i8 ? sycl::malloc_device<uint8_t>(expert_layout.size() * expert_slots, queue) : nullptr;
    const uint8_t ** d_ref_ptrs = predecoded_i8 ? sycl::malloc_device<const uint8_t *>(expert_slots, queue) : nullptr;
    uint8_t *        d_act      = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids      = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out      = sycl::malloc_device<float>(out_count, queue);
    float *          d_ref_out  = needs_ref ? sycl::malloc_device<float>(out_count, queue) : nullptr;

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
            std::copy(expert_layout.begin(), expert_layout.end(),
                      ref_weight_slices.begin() + slot * expert_layout.size());
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
            ref_args.output                             = d_ref_out;
            ref_args.expert_ptrs =
                predecoded_i8 ? reinterpret_cast<const void * const *>(d_ref_ptrs) : args.expert_ptrs;
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
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<int8_t, N> mxfp4_code_values_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> codes) {
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

template <int N>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<float, N> mxfp4_e8m0_to_fp32_prod_esimd(
    sycl::ext::intel::esimd::simd<uint8_t, N> e) {
    using namespace sycl::ext::intel::esimd;
    simd<uint32_t, N> bits = (convert<uint32_t>(e) - uint32_t{ 1 }) << 23;
    bits.merge(simd<uint32_t, N>(0x00000000u), e == uint8_t{ 0 });
    bits.merge(simd<uint32_t, N>(0x33800000u), e == uint8_t{ 1 });
    return bits.template bit_cast_view<float>();
}

template <int Repeat>
SYCL_ESIMD_FUNCTION inline void mxfp4_xmx_tiled_load_a_vec_from_group_local(
    const uint8_t *                                                             group,
    int64_t                                                                     tile_n_total,
    int64_t                                                                     xmx_row_in_group,
    sycl::ext::intel::esimd::simd<int8_t, Repeat * GGML_SYCL_MXFP4_MOE_XMX_K> & a_vec,
    sycl::ext::intel::esimd::simd<float, Repeat> &                              w_scale_vec) {
    using namespace sycl::ext::intel::esimd;
    constexpr int k_per         = GGML_SYCL_MXFP4_MOE_XMX_K;
    constexpr int packed_bytes  = k_per / 2;
    constexpr int compact_bytes = Repeat * packed_bytes;

    const uint8_t * scale_ptr  = group + xmx_row_in_group;
    const uint8_t * packed_ptr = group + tile_n_total + xmx_row_in_group * packed_bytes;

    simd<uint8_t, Repeat>        scale_bytes = block_load<uint8_t, Repeat>(scale_ptr);
    simd<uint8_t, compact_bytes> packed      = block_load<uint8_t, compact_bytes>(packed_ptr);
    w_scale_vec                              = mxfp4_e8m0_to_fp32_prod_esimd<Repeat>(scale_bytes);
#    pragma unroll
    for (int r = 0; r < Repeat; ++r) {
        simd<uint8_t, packed_bytes> row = packed.template select<packed_bytes, 1>(r * packed_bytes);
        simd<uint8_t, k_per>        codes;
        codes.template select<packed_bytes, 1>(0)            = row & uint8_t{ 0x0f };
        codes.template select<packed_bytes, 1>(packed_bytes) = row >> 4;
        a_vec.template select<k_per, 1>(r * k_per)           = mxfp4_code_values_esimd<k_per>(codes);
    }
}

SYCL_ESIMD_FUNCTION inline float mxfp4_e8m0_to_fp32_half_esimd_local(uint8_t e) {
    uint32_t bits = e == 0 ? 0x00400000u : (static_cast<uint32_t>(e - 1) << 23);
    float    result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
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

                simd<int, Repeat * ExecN> acc0   = 0;
                simd<int, Repeat * ExecN> acc1   = 0;
                simd<int, Repeat * ExecN> acc2   = 0;
                simd<int, Repeat * ExecN> acc3   = 0;
                const uint8_t *           a_ptr  = compact_a + (tile_m * k_tiles) * CompactABytes;
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

#if DPAS_EXPLORATION_ESIMD_AVAILABLE
template <int Repeat, int NTileRepeats, bool XmxLayout> struct mxfp4_dpas_compact_bytescale_kernel;

template <int Repeat, int NTileRepeats, bool XmxLayout>
static bool launch_mxfp4_dpas_compact_bytescale_kernel(const uint8_t * compact_a,
                                                       const int8_t *  b_base,
                                                       const float *   y_scales,
                                                       float *         c_base,
                                                       int64_t         m,
                                                       int64_t         n,
                                                       int64_t         k,
                                                       int64_t         tile_n_total,
                                                       int64_t         n_tile_groups_n,
                                                       sycl::queue &   queue,
                                                       std::string &   error) {
    static_assert(NTileRepeats == 1 || NTileRepeats == 2 || NTileRepeats == 4);
    constexpr int ExecN         = 16;
    constexpr int KPer          = 32;
    constexpr int AN            = Repeat * KPer;
    constexpr int BN            = KPer * ExecN;
    constexpr int ScaleBytes    = Repeat;
    constexpr int CompactABytes = Repeat * (KPer / 2);
    constexpr int ATileBytes    = ScaleBytes + CompactABytes;
    if (!compact_a || !b_base || !y_scales || !c_base) {
        error = "mxfp4_dpas_compact_bytescale buffers are null.";
        return false;
    }
    if ((m % Repeat) != 0 || (n % ExecN) != 0 || (k % KPer) != 0) {
        error = "mxfp4_dpas_compact_bytescale dims must be multiples of repeat, 16, and 32.";
        return false;
    }
    if constexpr (XmxLayout) {
        if (tile_n_total < Repeat || (tile_n_total % Repeat) != 0 || n_tile_groups_n <= 0) {
            error = "mxfp4_dpas_compact_bytescale XMX layout has invalid tile metadata.";
            return false;
        }
    }

    const int64_t m_tiles       = m / Repeat;
    const int64_t n_tiles       = n / ExecN;
    const int64_t n_tile_groups = n_tiles / NTileRepeats;
    const int64_t k_tiles       = k / KPer;
    const int64_t tiles         = m_tiles * n_tile_groups;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mxfp4_dpas_compact_bytescale_kernel<Repeat, NTileRepeats, XmxLayout>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(tiles)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL DPAS_NUM_REGS(128) {
                using namespace sycl::ext::intel::esimd;
                const int64_t tile_idx     = static_cast<int64_t>(item.get_global_id(0));
                const int64_t tile_m       = tile_idx / n_tile_groups;
                const int64_t tile_group_n = tile_idx - tile_m * n_tile_groups;
                const int64_t tile_n_base  = tile_group_n * NTileRepeats;

                simd<float, Repeat * ExecN> acc0            = 0.0f;
                simd<float, Repeat * ExecN> acc1            = 0.0f;
                simd<float, Repeat * ExecN> acc2            = 0.0f;
                simd<float, Repeat * ExecN> acc3            = 0.0f;
                const uint8_t *             a_ptr           = compact_a + (tile_m * k_tiles) * ATileBytes;
                const int64_t               xmx_group_bytes = tile_n_total * (1 + KPer / 2);
                const int64_t               xmx_row_start   = tile_m * Repeat;
                const int64_t               xmx_group_n     = tile_n_total > 0 ? xmx_row_start / tile_n_total : 0;
                const int64_t  xmx_row_in_group = tile_n_total > 0 ? xmx_row_start - xmx_group_n * tile_n_total : 0;
                const int8_t * b_ptr0           = b_base + ((tile_n_base + 0) * k_tiles) * BN;
                const int8_t * b_ptr1           = b_base + ((tile_n_base + (NTileRepeats >= 2 ? 1 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr2           = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 2 : 0)) * k_tiles) * BN;
                const int8_t * b_ptr3           = b_base + ((tile_n_base + (NTileRepeats >= 4 ? 3 : 0)) * k_tiles) * BN;

                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    const uint8_t * scale_ptr  = a_ptr;
                    const uint8_t * packed_ptr = a_ptr + ScaleBytes;
                    if constexpr (XmxLayout) {
                        const uint8_t * group = compact_a + (kt * n_tile_groups_n + xmx_group_n) * xmx_group_bytes;
                        scale_ptr             = group + xmx_row_in_group;
                        packed_ptr            = group + tile_n_total + xmx_row_in_group * (KPer / 2);
                    }
                    simd<uint8_t, ScaleBytes> scale_bytes =
                        dpas_block_load<uint8_t, ScaleBytes, DpasMemoryPattern::DIRECT_GLOBAL>(scale_ptr, false);
                    simd<uint8_t, CompactABytes> packed =
                        dpas_block_load<uint8_t, CompactABytes, DpasMemoryPattern::DIRECT_GLOBAL>(packed_ptr, false);
                    simd<int8_t, AN>    a_vec;
                    simd<float, Repeat> w_scale_vec;
#    pragma unroll
                    for (int r = 0; r < Repeat; ++r) {
                        simd<uint8_t, KPer / 2> row = packed.template select<KPer / 2, 1>(r * (KPer / 2));
                        a_vec.template select<KPer / 2, 1>(r * KPer) =
                            mxfp4_code_values_esimd<KPer / 2>(row & uint8_t{ 0x0f });
                        a_vec.template select<KPer / 2, 1>(r * KPer + KPer / 2) =
                            mxfp4_code_values_esimd<KPer / 2>(row >> 4);
                        w_scale_vec[r] = mxfp4_e8m0_to_fp32_half_esimd_local(scale_bytes[r]);
                    }

                    simd<int8_t, BN> b_vec0 =
                        dpas_block_load<int8_t, BN, DpasMemoryPattern::DIRECT_GLOBAL>(b_ptr0, false);
                    simd<int, Repeat * ExecN> part0 = 0;
                    part0 = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(part0, b_vec0, a_vec);

                    simd<float, ExecN> y_scale_vec0 =
                        block_load<float, ExecN>(y_scales + ((tile_n_base + 0) * k_tiles + kt) * ExecN);
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
                        simd<float, ExecN> y_scale_vec1 =
                            block_load<float, ExecN>(y_scales + ((tile_n_base + 1) * k_tiles + kt) * ExecN);
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
                        simd<float, ExecN> y_scale_vec2 =
                            block_load<float, ExecN>(y_scales + ((tile_n_base + 2) * k_tiles + kt) * ExecN);
                        simd<float, ExecN> y_scale_vec3 =
                            block_load<float, ExecN>(y_scales + ((tile_n_base + 3) * k_tiles + kt) * ExecN);
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

                    if constexpr (!XmxLayout) {
                        a_ptr += ATileBytes;
                    }
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
template <int Repeat, int NTileRepeats, bool XmxLayout>
static bool launch_mxfp4_dpas_compact_bytescale_kernel(const uint8_t *,
                                                       const int8_t *,
                                                       const float *,
                                                       float *,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       sycl::queue &,
                                                       std::string & error) {
    (void) Repeat;
    (void) NTileRepeats;
    (void) XmxLayout;
    error = "SYCL ESIMD unavailable; mxfp4_dpas_compact_bytescale disabled.";
    return false;
}
#endif

#if DPAS_EXPLORATION_ESIMD_AVAILABLE
template <int Repeat>
SYCL_ESIMD_FUNCTION inline void mxfp4_selected_xmx_dpas_accum_one(
    const uint8_t *                                                                                      gate_base,
    const uint8_t *                                                                                      up_base,
    const sycl::ext::intel::esimd::simd<int8_t, GGML_SYCL_MXFP4_MOE_XMX_K * GGML_SYCL_MXFP4_MOE_XMX_N> & b_vec,
    sycl::ext::intel::esimd::simd<float, 1>                                                              y_scale,
    int64_t                                                                                              kt,
    int64_t                                        n_tile_groups_n,
    int64_t                                        group_bytes,
    int64_t                                        tile_n_total,
    int64_t                                        xmx_group_n,
    int64_t                                        xmx_row_in_group,
    sycl::ext::intel::esimd::simd<float, Repeat> & gate_acc,
    sycl::ext::intel::esimd::simd<float, Repeat> & up_acc) {
    using namespace sycl::ext::intel::esimd;
    constexpr int exec_n = GGML_SYCL_MXFP4_MOE_XMX_N;
    constexpr int k_per  = GGML_SYCL_MXFP4_MOE_XMX_K;
    constexpr int an     = Repeat * k_per;

    const uint8_t * gate_group = gate_base + (kt * n_tile_groups_n + xmx_group_n) * group_bytes;
    const uint8_t * up_group   = up_base + (kt * n_tile_groups_n + xmx_group_n) * group_bytes;

    simd<int8_t, an>    gate_a_vec;
    simd<int8_t, an>    up_a_vec;
    simd<float, Repeat> gate_w_scale;
    simd<float, Repeat> up_w_scale;
    mxfp4_xmx_tiled_load_a_vec_from_group_local<Repeat>(gate_group, tile_n_total, xmx_row_in_group, gate_a_vec,
                                                        gate_w_scale);
    mxfp4_xmx_tiled_load_a_vec_from_group_local<Repeat>(up_group, tile_n_total, xmx_row_in_group, up_a_vec, up_w_scale);

    simd<int, Repeat * exec_n> gate_part = 0;
    simd<int, Repeat * exec_n> up_part   = 0;
    gate_part                            = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(gate_part, b_vec, gate_a_vec);
    up_part                              = xmx::dpas<8, Repeat, int, int, int8_t, int8_t>(up_part, b_vec, up_a_vec);
#    pragma unroll
    for (int r = 0; r < Repeat; ++r) {
        simd<int, 1>   gate_i = gate_part.template select<1, 1>(r * exec_n);
        simd<int, 1>   up_i   = up_part.template select<1, 1>(r * exec_n);
        simd<float, 1> gate_f = convert<float>(gate_i) * (y_scale * gate_w_scale[r]);
        simd<float, 1> up_f   = convert<float>(up_i) * (y_scale * up_w_scale[r]);
        gate_acc[r] += gate_f[0];
        up_acc[r] += up_f[0];
    }
}

SYCL_ESIMD_FUNCTION inline float mxfp4_selected_xmx_oai_glu_esimd(float gate, float up) {
    using namespace sycl::ext::intel::esimd;
    constexpr float alpha        = 1.702f;
    constexpr float limit        = 7.0f;
    const float     gate_limited = gate < limit ? gate : limit;
    float           up_limited   = up < limit ? up : limit;
    up_limited                   = up_limited > -limit ? up_limited : -limit;
    return (gate_limited / (1.0f + exp(-gate_limited * alpha))) * (1.0f + up_limited);
}

template <int Repeat>
SYCL_ESIMD_FUNCTION inline void mxfp4_selected_xmx_store_one(float * dst,
                                                             int64_t nrows_per_expert,
                                                             int64_t selected_slot,
                                                             int64_t tile_m,
                                                             sycl::ext::intel::esimd::simd<float, Repeat> gate_acc,
                                                             sycl::ext::intel::esimd::simd<float, Repeat> up_acc) {
    using namespace sycl::ext::intel::esimd;
#    pragma unroll
    for (int r = 0; r < Repeat; ++r) {
        const int row = static_cast<int>(tile_m) * Repeat + r;
        if (row < nrows_per_expert) {
            const float value = mxfp4_selected_xmx_oai_glu_esimd(gate_acc[r], up_acc[r]);
            block_store<float, 1>(dst + selected_slot * nrows_per_expert + row, simd<float, 1>(value));
        }
    }
}

template <int Repeat, bool M2> struct mxfp4_selected_xmx_dpas_tile_kernel;

template <int Repeat, bool M2>
static bool launch_mxfp4_selected_xmx_dpas_tile_kernel(const uint8_t * selected_tiles,
                                                       const uint8_t * q8_src,
                                                       float *         dst,
                                                       int64_t         nrows_per_expert,
                                                       int64_t         n_selected,
                                                       int64_t         ncols,
                                                       int64_t         ncols_y,
                                                       int64_t         expert_bytes,
                                                       int64_t         tile_n_total,
                                                       sycl::queue &   queue,
                                                       std::string &   error) {
    constexpr int exec_n = GGML_SYCL_MXFP4_MOE_XMX_N;
    constexpr int k_per  = GGML_SYCL_MXFP4_MOE_XMX_K;
    constexpr int bn     = k_per * exec_n;

    if (!selected_tiles || !q8_src || !dst) {
        error = "mxfp4_selected_xmx_dpas_tile buffers are null.";
        return false;
    }
    if (nrows_per_expert <= 0 || n_selected <= 0 || n_selected > 4 || ncols <= 0 || (ncols % k_per) != 0 ||
        tile_n_total < Repeat || (tile_n_total % Repeat) != 0 || expert_bytes <= 0) {
        error = "mxfp4_selected_xmx_dpas_tile received invalid dimensions.";
        return false;
    }

    const int64_t m_tiles         = (nrows_per_expert + Repeat - 1) / Repeat;
    const int64_t work_m_tiles    = M2 ? (m_tiles + 1) / 2 : m_tiles;
    const int64_t k_tiles         = ncols / k_per;
    const int64_t n_tile_groups_n = (nrows_per_expert + tile_n_total - 1) / tile_n_total;
    const int64_t group_bytes     = tile_n_total * (1 + k_per / 2);

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mxfp4_selected_xmx_dpas_tile_kernel<Repeat, M2>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(work_m_tiles)), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL DPAS_NUM_REGS(256) {
                using namespace sycl::ext::intel::esimd;
                const int64_t work_tile = static_cast<int64_t>(item.get_global_id(0));
                const int64_t tile_m0   = M2 ? work_tile * 2 : work_tile;
                const int64_t tile_m1   = tile_m0 + 1;
                const bool    have_m1   = M2 && tile_m1 < m_tiles;

                const int64_t xmx_row_start0    = tile_m0 * Repeat;
                const int64_t xmx_group_n0      = xmx_row_start0 / tile_n_total;
                const int64_t xmx_row_in_group0 = xmx_row_start0 - xmx_group_n0 * tile_n_total;
                const int64_t xmx_row_start1    = tile_m1 * Repeat;
                const int64_t xmx_group_n1      = xmx_row_start1 / tile_n_total;
                const int64_t xmx_row_in_group1 = xmx_row_start1 - xmx_group_n1 * tile_n_total;

                const uint8_t * gate_base0 = selected_tiles + 0 * 2 * expert_bytes;
                const uint8_t * up_base0   = gate_base0 + expert_bytes;
                const uint8_t * gate_base1 = selected_tiles + 1 * 2 * expert_bytes;
                const uint8_t * up_base1   = gate_base1 + expert_bytes;
                const uint8_t * gate_base2 = selected_tiles + 2 * 2 * expert_bytes;
                const uint8_t * up_base2   = gate_base2 + expert_bytes;
                const uint8_t * gate_base3 = selected_tiles + 3 * 2 * expert_bytes;
                const uint8_t * up_base3   = gate_base3 + expert_bytes;

                simd<float, Repeat> gate_acc0_s0 = 0.0f;
                simd<float, Repeat> up_acc0_s0   = 0.0f;
                simd<float, Repeat> gate_acc0_s1 = 0.0f;
                simd<float, Repeat> up_acc0_s1   = 0.0f;
                simd<float, Repeat> gate_acc0_s2 = 0.0f;
                simd<float, Repeat> up_acc0_s2   = 0.0f;
                simd<float, Repeat> gate_acc0_s3 = 0.0f;
                simd<float, Repeat> up_acc0_s3   = 0.0f;
                simd<float, Repeat> gate_acc1_s0 = 0.0f;
                simd<float, Repeat> up_acc1_s0   = 0.0f;
                simd<float, Repeat> gate_acc1_s1 = 0.0f;
                simd<float, Repeat> up_acc1_s1   = 0.0f;
                simd<float, Repeat> gate_acc1_s2 = 0.0f;
                simd<float, Repeat> up_acc1_s2   = 0.0f;
                simd<float, Repeat> gate_acc1_s3 = 0.0f;
                simd<float, Repeat> up_acc1_s3   = 0.0f;

                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    simd<int8_t, k_per> q_vec =
                        block_load<int8_t, k_per>(reinterpret_cast<const int8_t *>(q8_src) + kt * k_per);
                    simd<int8_t, bn> b_vec = 0;
#    pragma unroll
                    for (int kk = 0; kk < k_per; ++kk) {
                        b_vec[(kk / 4) * exec_n * 4 + (kk % 4)] = q_vec[kk];
                    }

                    const sycl::half * y_scale_ptr =
                        reinterpret_cast<const sycl::half *>(q8_src + ncols_y + kt * 2 * sizeof(sycl::half));
                    simd<sycl::half, 1> y_half  = block_load<sycl::half, 1>(y_scale_ptr);
                    simd<float, 1>      y_scale = y_half;

                    if (n_selected > 0) {
                        mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                            gate_base0, up_base0, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                            xmx_group_n0, xmx_row_in_group0, gate_acc0_s0, up_acc0_s0);
                        if (have_m1) {
                            mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                                gate_base0, up_base0, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                                xmx_group_n1, xmx_row_in_group1, gate_acc1_s0, up_acc1_s0);
                        }
                    }
                    if (n_selected > 1) {
                        mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                            gate_base1, up_base1, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                            xmx_group_n0, xmx_row_in_group0, gate_acc0_s1, up_acc0_s1);
                        if (have_m1) {
                            mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                                gate_base1, up_base1, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                                xmx_group_n1, xmx_row_in_group1, gate_acc1_s1, up_acc1_s1);
                        }
                    }
                    if (n_selected > 2) {
                        mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                            gate_base2, up_base2, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                            xmx_group_n0, xmx_row_in_group0, gate_acc0_s2, up_acc0_s2);
                        if (have_m1) {
                            mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                                gate_base2, up_base2, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                                xmx_group_n1, xmx_row_in_group1, gate_acc1_s2, up_acc1_s2);
                        }
                    }
                    if (n_selected > 3) {
                        mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                            gate_base3, up_base3, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                            xmx_group_n0, xmx_row_in_group0, gate_acc0_s3, up_acc0_s3);
                        if (have_m1) {
                            mxfp4_selected_xmx_dpas_accum_one<Repeat>(
                                gate_base3, up_base3, b_vec, y_scale, kt, n_tile_groups_n, group_bytes, tile_n_total,
                                xmx_group_n1, xmx_row_in_group1, gate_acc1_s3, up_acc1_s3);
                        }
                    }
                }

                if (n_selected > 0) {
                    mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 0, tile_m0, gate_acc0_s0, up_acc0_s0);
                    if (have_m1) {
                        mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 0, tile_m1, gate_acc1_s0,
                                                             up_acc1_s0);
                    }
                }
                if (n_selected > 1) {
                    mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 1, tile_m0, gate_acc0_s1, up_acc0_s1);
                    if (have_m1) {
                        mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 1, tile_m1, gate_acc1_s1,
                                                             up_acc1_s1);
                    }
                }
                if (n_selected > 2) {
                    mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 2, tile_m0, gate_acc0_s2, up_acc0_s2);
                    if (have_m1) {
                        mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 2, tile_m1, gate_acc1_s2,
                                                             up_acc1_s2);
                    }
                }
                if (n_selected > 3) {
                    mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 3, tile_m0, gate_acc0_s3, up_acc0_s3);
                    if (have_m1) {
                        mxfp4_selected_xmx_store_one<Repeat>(dst, nrows_per_expert, 3, tile_m1, gate_acc1_s3,
                                                             up_acc1_s3);
                    }
                }
            });
    });
    ev.wait_and_throw();
    return true;
}
#else
template <int Repeat, bool M2>
static bool launch_mxfp4_selected_xmx_dpas_tile_kernel(const uint8_t *,
                                                       const uint8_t *,
                                                       float *,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       int64_t,
                                                       sycl::queue &,
                                                       std::string & error) {
    (void) Repeat;
    (void) M2;
    error = "SYCL ESIMD unavailable; mxfp4_selected_xmx_dpas_tile disabled.";
    return false;
}
#endif

bool run_mxfp4_selected_xmx_dpas_tile(const GeneratedWeights &     weights,
                                      const GeneratedActivations & activations,
                                      int64_t                      m,
                                      int64_t                      n_selected,
                                      int64_t                      k,
                                      int                          requested_tiles_n,
                                      int                          m_tiles_per_work_item,
                                      bool                         validate,
                                      int                          warmup,
                                      int                          iterations,
                                      sycl::queue &                queue,
                                      ReferenceMetrics &           out,
                                      std::string &                error) {
    if (m <= 0 || n_selected <= 0 || n_selected > 4 || k <= 0 || (k % QK_MXFP4) != 0 || (k % QK8_1) != 0) {
        error = "mxfp4_selected_xmx_dpas_tile requires positive M/K, 1..4 selected routes, and block-aligned K.";
        return false;
    }
    if (m_tiles_per_work_item != 1 && m_tiles_per_work_item != 2) {
        error = "mxfp4_selected_xmx_dpas_tile supports one or two M tiles per work-item.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_selected_xmx_dpas_tile requires SOA MXFP4 source weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_selected_xmx_dpas_tile requires SOA Q8_1 activations.";
        return false;
    }

    int tiles_n = 0;
    if (!select_mxfp4_xmx_tiles_n(queue, requested_tiles_n, tiles_n, error)) {
        return false;
    }
    const int tile_n_total = tiles_n * static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N);

    std::vector<uint8_t> xmx_layout;
    if (!make_mxfp4_xmx_tiled_layout(weights.layout, m, k, tile_n_total, xmx_layout, error)) {
        return false;
    }

    constexpr size_t selected_slots       = 4;
    constexpr size_t roles                = 2;
    const size_t     selected_count       = static_cast<size_t>(n_selected);
    const size_t     rows                 = static_cast<size_t>(m);
    const size_t     expert_bytes         = xmx_layout.size();
    const size_t     logical_expert_bytes = weights.layout.size();
    const size_t     weight_bytes         = selected_slots * roles * expert_bytes;
    const size_t     act_bytes            = activations.q8_1.size();
    const size_t     out_count            = rows * selected_count;
    const size_t     out_bytes            = out_count * sizeof(float);

    std::vector<uint8_t> selected_tiles(weight_bytes);
    for (size_t sel = 0; sel < selected_slots; ++sel) {
        for (size_t role = 0; role < roles; ++role) {
            std::copy(xmx_layout.begin(), xmx_layout.end(),
                      selected_tiles.begin() + (sel * roles + role) * expert_bytes);
        }
    }

    uint8_t * d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t * d_act     = sycl::malloc_device<uint8_t>(act_bytes, queue);
    float *   d_out     = sycl::malloc_device<float>(out_count, queue);

    auto cleanup = [&]() {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_act) {
            sycl::free(d_act, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
    };

    if (!d_weights || !d_act || !d_out) {
        cleanup();
        error = "device allocation failed for mxfp4_selected_xmx_dpas_tile.";
        return false;
    }

    queue.memcpy(d_weights, selected_tiles.data(), weight_bytes);
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    auto launch = [&]() {
        if (m_tiles_per_work_item == 2) {
            return launch_mxfp4_selected_xmx_dpas_tile_kernel<8, true>(d_weights, d_act, d_out, m, n_selected, k, k,
                                                                       static_cast<int64_t>(expert_bytes), tile_n_total,
                                                                       queue, error);
        }
        return launch_mxfp4_selected_xmx_dpas_tile_kernel<8, false>(d_weights, d_act, d_out, m, n_selected, k, k,
                                                                    static_cast<int64_t>(expert_bytes), tile_n_total,
                                                                    queue, error);
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
    const double ops      = 4.0 * static_cast<double>(m) * static_cast<double>(n_selected) * static_cast<double>(k);
    const double bytes    = 2.0 * static_cast<double>(logical_expert_bytes) * static_cast<double>(selected_count) +
                         static_cast<double>(act_bytes) + static_cast<double>(out_bytes);

    out.total_us       = mean_us;
    out.gemm_us        = mean_us;
    out.tflops         = mean_s > 0.0 ? (ops / mean_s) / 1.0e12 : 0.0;
    out.bandwidth_gbps = mean_s > 0.0 ? (bytes / mean_s) / 1.0e9 : 0.0;

    if (validate) {
        std::vector<float> actual(out_count);
        queue.memcpy(actual.data(), d_out, out_bytes).wait();

        const int64_t   blocks_per_row = k / QK_MXFP4;
        const size_t    qs_bytes       = rows * static_cast<size_t>(blocks_per_row) * static_cast<size_t>(QK_MXFP4 / 2);
        const uint8_t * src_qs         = weights.layout.data();
        const uint8_t * src_scales     = weights.layout.data() + qs_bytes;
        const uint8_t * q8_row         = activations.q8_1.data();

        double max_err = 0.0;
        double sum_err = 0.0;
        for (size_t sel = 0; sel < selected_count; ++sel) {
            for (size_t row = 0; row < rows; ++row) {
                float dot = 0.0f;
                for (int64_t block = 0; block < blocks_per_row; ++block) {
                    const size_t    block_idx = row * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(block);
                    const uint8_t * qs        = src_qs + block_idx * static_cast<size_t>(QK_MXFP4 / 2);
                    const int8_t *  q8_qs =
                        reinterpret_cast<const int8_t *>(q8_row) + static_cast<size_t>(block) * QK8_1;
                    const float scale = mxfp4_e8m0_to_fp32_prod_host(src_scales[block_idx]) *
                                        mxfp4_selected_kmajor_q8_1_scale(q8_row, k, block);
                    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                        const uint8_t packed = qs[i];
                        dot += scale * static_cast<float>(mxfp4_code_value(packed & 0x0f)) *
                               static_cast<float>(q8_qs[static_cast<size_t>(i)]);
                        dot += scale * static_cast<float>(mxfp4_code_value(packed >> 4)) *
                               static_cast<float>(q8_qs[static_cast<size_t>(i + QK_MXFP4 / 2)]);
                    }
                }

                constexpr float alpha        = 1.702f;
                constexpr float limit        = 7.0f;
                const float     gate_limited = std::fmin(dot, limit);
                const float     up_limited   = std::fmax(std::fmin(dot, limit), -limit);
                const float  expected = (gate_limited / (1.0f + std::exp(-gate_limited * alpha))) * (1.0f + up_limited);
                const size_t idx      = sel * rows + row;
                const double diff     = std::fabs(static_cast<double>(actual[idx]) - static_cast<double>(expected));
                max_err               = std::max(max_err, diff);
                sum_err += diff;
                const double tol = 2e-2 + 2e-2 * std::fabs(static_cast<double>(expected));
                if (!std::isfinite(actual[idx]) || diff > tol) {
                    cleanup();
                    char msg[320];
                    std::snprintf(msg, sizeof(msg),
                                  "mxfp4_selected_xmx_dpas_tile validation failed at %zu: actual=%.6f "
                                  "expected=%.6f diff=%.6f tol=%.6f max=%.6f mean=%.6f.",
                                  idx, static_cast<double>(actual[idx]), static_cast<double>(expected), diff, tol,
                                  max_err, sum_err / static_cast<double>(idx + 1));
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
template <int Repeat, int NTileRepeats, bool PackedScales> struct mxfp4_dpas_compact_scaled_kernel;

template <int Repeat, int NTileRepeats, bool PackedScales>
static bool launch_mxfp4_dpas_compact_scaled_kernel(const uint8_t * compact_a,
                                                    const int8_t *  b_base,
                                                    const float *   w_scales,
                                                    const float *   y_scales,
                                                    float *         c_base,
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
    if (!compact_a || !b_base || !w_scales || !y_scales || !c_base) {
        error = "mxfp4_dpas_compact_scaled buffers are null.";
        return false;
    }
    if ((m % Repeat) != 0 || (n % ExecN) != 0 || (k % KPer) != 0) {
        error = "mxfp4_dpas_compact_scaled dims must be multiples of repeat, 16, and 32.";
        return false;
    }

    const int64_t m_tiles       = m / Repeat;
    const int64_t n_tiles       = n / ExecN;
    const int64_t n_tile_groups = n_tiles / NTileRepeats;
    const int64_t k_tiles       = k / KPer;
    const int64_t tiles         = m_tiles * n_tile_groups;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mxfp4_dpas_compact_scaled_kernel<Repeat, NTileRepeats, PackedScales>>(
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
                const uint8_t *             a_ptr  = compact_a + (tile_m * k_tiles) * CompactABytes;
                const int8_t *              b_ptr0 = b_base + ((tile_n_base + 0) * k_tiles) * BN;
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

                    a_ptr += CompactABytes;
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
static bool launch_mxfp4_dpas_compact_scaled_kernel(const uint8_t *,
                                                    const int8_t *,
                                                    const float *,
                                                    const float *,
                                                    float *,
                                                    int64_t,
                                                    int64_t,
                                                    int64_t,
                                                    sycl::queue &,
                                                    std::string & error) {
    (void) Repeat;
    (void) NTileRepeats;
    (void) PackedScales;
    error = "SYCL ESIMD unavailable; mxfp4_dpas_compact_scaled disabled.";
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

bool run_mxfp4_dpas_grouped_compact_scaled(const GeneratedWeights &     weights,
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
        error = "mxfp4_dpas_compact_scaled requires M multiple of 8, N multiple of 16, and K multiple of 32.";
        return false;
    }
    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4) {
        error = "mxfp4_dpas_compact_scaled n-tile repeats must be 1, 2, or 4.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_dpas_compact_scaled requires SOA MXFP4 weights as the source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_dpas_compact_scaled requires SOA Q8_1 activations.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const int64_t m_tiles        = m / repeat;
    const int64_t n_tiles        = n / exec_n;
    const int64_t k_tiles        = k / k_per;
    if ((n_tiles % n_tile_repeats) != 0) {
        error = "mxfp4_dpas_compact_scaled dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }

    const size_t nblocks       = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t qs_bytes      = nblocks * (QK_MXFP4 / 2);
    const size_t scale_bytes   = nblocks;
    const size_t q8_row_bytes  = static_cast<size_t>(k / QK8_1) * sizeof(block_q8_1);
    const size_t a_tile_bytes  = static_cast<size_t>(repeat) * static_cast<size_t>(k_per / 2);
    const size_t b_tile_elems  = static_cast<size_t>(k_per) * static_cast<size_t>(exec_n);
    const size_t a_bytes       = static_cast<size_t>(m_tiles) * static_cast<size_t>(k_tiles) * a_tile_bytes;
    const size_t b_elems       = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t c_elems       = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t b_bytes       = b_elems * sizeof(int8_t);
    const size_t c_bytes       = c_elems * sizeof(float);
    const size_t w_scale_count = static_cast<size_t>(m) * static_cast<size_t>(k_tiles);
    const size_t y_scale_count = static_cast<size_t>(n) * static_cast<size_t>(k_tiles);
    const size_t w_scale_bytes = w_scale_count * sizeof(float);
    const size_t y_scale_bytes = y_scale_count * sizeof(float);
    if (weights.layout.size() < qs_bytes + scale_bytes) {
        error = "mxfp4_dpas_compact_scaled source weight layout is too small.";
        return false;
    }
    if (activations.q8_1.size() < static_cast<size_t>(n) * q8_row_bytes) {
        error = "mxfp4_dpas_compact_scaled activation buffer is too small.";
        return false;
    }

    std::vector<uint8_t> a_host(a_bytes, 0);
    std::vector<int8_t>  b_host(b_elems, 0);
    std::vector<float>   w_scale_host(w_scale_count, 0.0f);
    std::vector<float>   y_scale_host(y_scale_count, 0.0f);
    std::vector<int8_t>  weight_raw;
    std::vector<int8_t>  act_raw;
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

    uint8_t * d_a       = sycl::malloc_device<uint8_t>(a_bytes, queue);
    int8_t *  d_b       = sycl::malloc_device<int8_t>(b_elems, queue);
    float *   d_w_scale = sycl::malloc_device<float>(w_scale_count, queue);
    float *   d_y_scale = sycl::malloc_device<float>(y_scale_count, queue);
    float *   d_c       = sycl::malloc_device<float>(c_elems, queue);
    auto      cleanup   = [&]() {
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
        error = "device allocation failed for mxfp4_dpas_compact_scaled.";
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
                    return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 1, true>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 1, false>(d_a, d_b, d_w_scale, d_y_scale, d_c, m,
                                                                                 n, k, queue, error);
            case 2:
                if (packed_scales) {
                    return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 2, true>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 2, false>(d_a, d_b, d_w_scale, d_y_scale, d_c, m,
                                                                                 n, k, queue, error);
            case 4:
                if (packed_scales) {
                    return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 4, true>(d_a, d_b, d_w_scale, d_y_scale, d_c,
                                                                                    m, n, k, queue, error);
                }
                return launch_mxfp4_dpas_compact_scaled_kernel<repeat, 4, false>(d_a, d_b, d_w_scale, d_y_scale, d_c, m,
                                                                                 n, k, queue, error);
            default:
                error = "mxfp4_dpas_compact_scaled unsupported n-tile repeats.";
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
                    oss << "mxfp4_dpas_compact_scaled validation failed at row=" << row << " col=" << col
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

bool run_mxfp4_dpas_grouped_compact_bytescale(const GeneratedWeights &     weights,
                                              const GeneratedActivations & activations,
                                              int64_t                      m,
                                              int64_t                      n,
                                              int64_t                      k,
                                              int                          n_tile_repeats,
                                              bool                         xmx_layout,
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
        error = "mxfp4_dpas_compact_bytescale requires M multiple of 8, N multiple of 16, and K multiple of 32.";
        return false;
    }
    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4) {
        error = "mxfp4_dpas_compact_bytescale n-tile repeats must be 1, 2, or 4.";
        return false;
    }
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_dpas_compact_bytescale requires SOA MXFP4 weights as the source layout.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_dpas_compact_bytescale requires SOA Q8_1 activations.";
        return false;
    }

    const int64_t blocks_per_row = k / QK_MXFP4;
    const int64_t m_tiles        = m / repeat;
    const int64_t n_tiles        = n / exec_n;
    const int64_t k_tiles        = k / k_per;
    if ((n_tiles % n_tile_repeats) != 0) {
        error = "mxfp4_dpas_compact_bytescale dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }

    const size_t nblocks      = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);
    const size_t qs_bytes     = nblocks * (QK_MXFP4 / 2);
    const size_t scale_bytes  = nblocks;
    const size_t q8_row_bytes = static_cast<size_t>(k / QK8_1) * sizeof(block_q8_1);
    if (weights.layout.size() < qs_bytes + scale_bytes) {
        error = "mxfp4_dpas_compact_bytescale source weight layout is too small.";
        return false;
    }
    if (activations.q8_1.size() < static_cast<size_t>(n) * q8_row_bytes) {
        error = "mxfp4_dpas_compact_bytescale activation buffer is too small.";
        return false;
    }

    int64_t tile_n_total = repeat;
    if (xmx_layout) {
        const int device_id = ggml_sycl_get_device_id_from_queue(queue);
        if (device_id >= 0 && device_id < ggml_sycl_info().device_count) {
            const auto & caps = ggml_sycl_info().devices[device_id].xmx_caps;
            if (caps.N > 0 && caps.optimal_tiles_n > 0) {
                tile_n_total = static_cast<int64_t>(caps.N) * static_cast<int64_t>(caps.optimal_tiles_n);
            }
        }
        if (tile_n_total < repeat || (tile_n_total % repeat) != 0) {
            error = "mxfp4_dpas_compact_bytescale invalid queried XMX tile width.";
            return false;
        }
    }
    const int64_t n_tile_groups_n = xmx_layout ? (m + tile_n_total - 1) / tile_n_total : m_tiles;
    const size_t  a_tile_bytes    = xmx_layout ? static_cast<size_t>(tile_n_total) * (1 + QK_MXFP4 / 2) :
                                                 static_cast<size_t>(repeat) + static_cast<size_t>(repeat) * (QK_MXFP4 / 2);
    const size_t  b_tile_elems    = static_cast<size_t>(k_per) * static_cast<size_t>(exec_n);
    const size_t  a_bytes         = static_cast<size_t>(n_tile_groups_n) * static_cast<size_t>(k_tiles) * a_tile_bytes;
    const size_t  b_elems         = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t  c_elems         = static_cast<size_t>(m) * static_cast<size_t>(n);
    const size_t  b_bytes         = b_elems * sizeof(int8_t);
    const size_t  c_bytes         = c_elems * sizeof(float);
    const size_t  y_scale_count =
        static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * static_cast<size_t>(exec_n);
    const size_t y_scale_bytes = y_scale_count * sizeof(float);

    std::vector<uint8_t> a_host(a_bytes, 0);
    std::vector<int8_t>  b_host(b_elems, 0);
    std::vector<float>   y_scale_host(y_scale_count, 0.0f);
    std::vector<int8_t>  weight_raw;
    std::vector<int8_t>  act_raw;
    if (validate) {
        weight_raw.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0);
        act_raw.assign(static_cast<size_t>(n) * static_cast<size_t>(k), 0);
    }

    const uint8_t * qs_base           = weights.layout.data();
    const uint8_t * scale_base        = weights.layout.data() + qs_bytes;
    auto            record_weight_raw = [&](int64_t row, int64_t kt, const uint8_t * block_qs) {
        if (!validate) {
            return;
        }
        for (int i = 0; i < QK_MXFP4 / 2; ++i) {
            const uint8_t packed = block_qs[i];
            const int64_t k0     = kt * k_per + i;
            const int64_t k1     = kt * k_per + (QK_MXFP4 / 2) + i;
            weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k0)] =
                mxfp4_code_value(static_cast<uint8_t>(packed & 0x0f));
            weight_raw[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(k1)] =
                mxfp4_code_value(static_cast<uint8_t>(packed >> 4));
        }
    };

    if (xmx_layout) {
        for (int64_t kt = 0; kt < k_tiles; ++kt) {
            for (int64_t tg_n = 0; tg_n < n_tile_groups_n; ++tg_n) {
                uint8_t * group = a_host.data() + (static_cast<size_t>(kt) * static_cast<size_t>(n_tile_groups_n) +
                                                   static_cast<size_t>(tg_n)) *
                                                      a_tile_bytes;
                uint8_t * scales = group;
                uint8_t * qs     = group + static_cast<size_t>(tile_n_total);
                for (int64_t tn = 0; tn < tile_n_total; ++tn) {
                    const int64_t row = tg_n * tile_n_total + tn;
                    if (row >= m) {
                        continue;
                    }
                    const size_t block_idx =
                        static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
                    const uint8_t * block_qs = qs_base + block_idx * (QK_MXFP4 / 2);
                    scales[tn]               = scale_base[block_idx];
                    std::copy(block_qs, block_qs + (QK_MXFP4 / 2), qs + static_cast<size_t>(tn) * (QK_MXFP4 / 2));
                    record_weight_raw(row, kt, block_qs);
                }
            }
        }
    } else {
        for (int64_t tile_m = 0; tile_m < m_tiles; ++tile_m) {
            for (int64_t kt = 0; kt < k_tiles; ++kt) {
                uint8_t * tile = a_host.data() + (static_cast<size_t>(tile_m) * static_cast<size_t>(k_tiles) +
                                                  static_cast<size_t>(kt)) *
                                                     a_tile_bytes;
                uint8_t * scales = tile;
                uint8_t * qs     = tile + static_cast<size_t>(repeat);
                for (int r = 0; r < repeat; ++r) {
                    const int64_t row = tile_m * repeat + r;
                    const size_t  block_idx =
                        static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
                    const uint8_t * block_qs = qs_base + block_idx * (QK_MXFP4 / 2);
                    scales[r]                = scale_base[block_idx];
                    std::copy(block_qs, block_qs + (QK_MXFP4 / 2), qs + static_cast<size_t>(r) * (QK_MXFP4 / 2));
                    record_weight_raw(row, kt, block_qs);
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
            y_scale_host[(static_cast<size_t>(row / exec_n) * static_cast<size_t>(k_tiles) + static_cast<size_t>(kt)) *
                             static_cast<size_t>(exec_n) +
                         static_cast<size_t>(row % exec_n)] = static_cast<float>(d);
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

    uint8_t * d_a       = sycl::malloc_device<uint8_t>(a_bytes, queue);
    int8_t *  d_b       = sycl::malloc_device<int8_t>(b_elems, queue);
    float *   d_y_scale = sycl::malloc_device<float>(y_scale_count, queue);
    float *   d_c       = sycl::malloc_device<float>(c_elems, queue);
    auto      cleanup   = [&]() {
        if (d_a) {
            sycl::free(d_a, queue);
        }
        if (d_b) {
            sycl::free(d_b, queue);
        }
        if (d_y_scale) {
            sycl::free(d_y_scale, queue);
        }
        if (d_c) {
            sycl::free(d_c, queue);
        }
    };
    if (!d_a || !d_b || !d_y_scale || !d_c) {
        cleanup();
        error = "device allocation failed for mxfp4_dpas_compact_bytescale.";
        return false;
    }

    queue.memcpy(d_a, a_host.data(), a_bytes);
    queue.memcpy(d_b, b_host.data(), b_bytes);
    queue.memcpy(d_y_scale, y_scale_host.data(), y_scale_bytes);
    queue.memset(d_c, 0, c_bytes);
    queue.wait_and_throw();

    auto launch = [&]() {
        switch (n_tile_repeats) {
            case 1:
                if (xmx_layout) {
                    return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 1, true>(
                        d_a, d_b, d_y_scale, d_c, m, n, k, tile_n_total, n_tile_groups_n, queue, error);
                }
                return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 1, false>(d_a, d_b, d_y_scale, d_c, m, n, k,
                                                                                    0, 0, queue, error);
            case 2:
                if (xmx_layout) {
                    return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 2, true>(
                        d_a, d_b, d_y_scale, d_c, m, n, k, tile_n_total, n_tile_groups_n, queue, error);
                }
                return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 2, false>(d_a, d_b, d_y_scale, d_c, m, n, k,
                                                                                    0, 0, queue, error);
            case 4:
                if (xmx_layout) {
                    return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 4, true>(
                        d_a, d_b, d_y_scale, d_c, m, n, k, tile_n_total, n_tile_groups_n, queue, error);
                }
                return launch_mxfp4_dpas_compact_bytescale_kernel<repeat, 4, false>(d_a, d_b, d_y_scale, d_c, m, n, k,
                                                                                    0, 0, queue, error);
            default:
                error = "mxfp4_dpas_compact_bytescale unsupported n-tile repeats.";
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
    const double bytes    = static_cast<double>(a_bytes + b_bytes + y_scale_bytes + c_bytes);
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
                    const size_t block_idx =
                        static_cast<size_t>(row) * static_cast<size_t>(blocks_per_row) + static_cast<size_t>(kt);
                    const float ws = mxfp4_e8m0_to_fp32_device(scale_base[block_idx]) * 0.5f;
                    const float ys = y_scale_host[(static_cast<size_t>(col / exec_n) * static_cast<size_t>(k_tiles) +
                                                   static_cast<size_t>(kt)) *
                                                      static_cast<size_t>(exec_n) +
                                                  static_cast<size_t>(col % exec_n)];
                    ref += static_cast<float>(dot) * ws * ys;
                }
                const size_t idx  = static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float  diff = std::fabs(actual[idx] - ref);
                const float  tol  = std::max(0.05f, std::fabs(ref) * 1.0e-3f);
                if (!std::isfinite(actual[idx]) || diff > tol) {
                    std::ostringstream oss;
                    oss << "mxfp4_dpas_compact_bytescale validation failed at row=" << row << " col=" << col
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

bool run_mxfp4_mmv_id_xmx_tiled(const GeneratedWeights &     weights,
                                const GeneratedActivations & activations,
                                int64_t                      m,
                                int64_t                      n_selected,
                                int64_t                      k,
                                int64_t                      n_tokens,
                                int                          requested_tiles_n,
                                bool                         sparse_expert_slots,
                                bool                         raw_accum,
                                bool                         i8_rowmajor,
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
    if (!i8_rowmajor && !select_mxfp4_xmx_tiles_n(queue, requested_tiles_n, tiles_n, error)) {
        return false;
    }
    const int tile_n_total = i8_rowmajor ? 0 : tiles_n * static_cast<int>(GGML_SYCL_MXFP4_MOE_XMX_N);

    std::vector<uint8_t> launch_layout;
    if (i8_rowmajor) {
        if (!make_mxfp4_predecoded_i8_layout(weights.layout, m, k, static_cast<int>(k / QK_MXFP4), launch_layout,
                                             error)) {
            return false;
        }
    } else {
        if (!make_mxfp4_xmx_tiled_layout(weights.layout, m, k, tile_n_total, launch_layout, error)) {
            return false;
        }
    }

    const size_t  selected_count       = static_cast<size_t>(n_selected);
    const size_t  token_count          = static_cast<size_t>(n_tokens);
    const size_t  ids_count            = selected_count * token_count;
    const size_t  launch_expert_bytes  = launch_layout.size();
    const size_t  logical_expert_bytes = weights.layout.size();
    const size_t  expert_slots         = sparse_expert_slots ? std::max<size_t>(32, selected_count) : selected_count;
    const size_t  launch_weight_bytes  = launch_expert_bytes * expert_slots;
    const size_t  act_bytes            = activations.q8_1.size();
    const size_t  out_count            = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes            = out_count * sizeof(float);
    const int64_t q8_row_bytes         = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);
    const bool    compare_to_soa       = validate && !raw_accum;
    const int64_t k_tiles              = k / GGML_SYCL_MXFP4_MOE_XMX_K;
    const size_t  dpas_b_bytes         = i8_rowmajor ?
                                             ids_count * static_cast<size_t>(k_tiles) *
                                        static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_K * GGML_SYCL_MXFP4_MOE_XMX_N) :
                                             0;
    const size_t  dpas_y_bytes         = i8_rowmajor ? ids_count * static_cast<size_t>(k_tiles) *
                                                  static_cast<size_t>(GGML_SYCL_MXFP4_MOE_XMX_N) * sizeof(float) :
                                                       0;

    std::vector<uint8_t> launch_slices(launch_weight_bytes);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        std::copy(launch_layout.begin(), launch_layout.end(), launch_slices.begin() + slot * launch_expert_bytes);
    }

    uint8_t *        d_tiled              = sycl::malloc_device<uint8_t>(launch_weight_bytes, queue);
    const uint8_t ** d_ptrs               = sycl::malloc_device<const uint8_t *>(expert_slots, queue);
    uint8_t *        d_act                = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids                = sycl::malloc_device<int32_t>(ids_count, queue);
    int32_t *        d_grouped_experts    = nullptr;
    int32_t *        d_grouped_offsets    = nullptr;
    int32_t *        d_grouped_rows       = nullptr;
    int32_t *        d_grouped_chunks     = nullptr;
    int32_t *        d_grouped_row_starts = nullptr;
    float *          d_out                = sycl::malloc_device<float>(out_count, queue);
    int8_t *         d_dpas_b =
        i8_rowmajor ? sycl::malloc_device<int8_t>(dpas_b_bytes == 0 ? 1 : dpas_b_bytes, queue) : nullptr;
    float *   d_dpas_y = i8_rowmajor ? sycl::malloc_device<float>(
                                         (dpas_y_bytes == 0 ? sizeof(float) : dpas_y_bytes) / sizeof(float), queue) :
                                       nullptr;
    uint8_t * d_ref_weights =
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
        if (d_grouped_experts) {
            sycl::free(d_grouped_experts, queue);
        }
        if (d_grouped_offsets) {
            sycl::free(d_grouped_offsets, queue);
        }
        if (d_grouped_rows) {
            sycl::free(d_grouped_rows, queue);
        }
        if (d_grouped_chunks) {
            sycl::free(d_grouped_chunks, queue);
        }
        if (d_grouped_row_starts) {
            sycl::free(d_grouped_row_starts, queue);
        }
        if (d_out) {
            sycl::free(d_out, queue);
        }
        if (d_dpas_b) {
            sycl::free(d_dpas_b, queue);
        }
        if (d_dpas_y) {
            sycl::free(d_dpas_y, queue);
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

    if (!d_tiled || !d_ptrs || !d_act || !d_ids || !d_out || (i8_rowmajor && (!d_dpas_b || !d_dpas_y)) ||
        (compare_to_soa && (!d_ref_weights || !d_ref_ptrs || !d_ref_out))) {
        cleanup();
        error = "device allocation failed for mxfp4_mmv_id_xmx_tiled.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(expert_slots);
    std::vector<const uint8_t *> host_ref_ptrs;
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t slot = 0; slot < expert_slots; ++slot) {
        host_ptrs[slot] = d_tiled + slot * launch_expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            const size_t slot = sparse_expert_slots ? sparse_expert_slot(sel, selected_count, expert_slots) : sel;
            host_ids[token * selected_count + sel] = static_cast<int32_t>(slot);
        }
    }

    std::vector<int32_t> grouped_experts_host;
    std::vector<int32_t> grouped_offsets_host;
    std::vector<int32_t> grouped_rows_host;
    std::vector<int32_t> grouped_chunks_host;
    std::vector<int32_t> grouped_row_starts_host;
    if (i8_rowmajor) {
        constexpr int                     exec_n = GGML_SYCL_MXFP4_MOE_XMX_N;
        std::vector<std::vector<int32_t>> grouped_slots;
        grouped_slots.reserve(std::min(expert_slots, ids_count));
        for (size_t token = 0; token < token_count; ++token) {
            for (size_t sel = 0; sel < selected_count; ++sel) {
                const int32_t eid = host_ids[token * selected_count + sel];
                auto          it  = std::find(grouped_experts_host.begin(), grouped_experts_host.end(), eid);
                size_t        group_index;
                if (it == grouped_experts_host.end()) {
                    group_index = grouped_experts_host.size();
                    grouped_experts_host.push_back(eid);
                    grouped_slots.emplace_back();
                } else {
                    group_index = static_cast<size_t>(std::distance(grouped_experts_host.begin(), it));
                }
                grouped_slots[group_index].push_back(static_cast<int32_t>(sel * token_count + token));
            }
        }
        grouped_offsets_host.reserve(grouped_experts_host.size() + 1);
        grouped_offsets_host.push_back(0);
        for (size_t group = 0; group < grouped_slots.size(); ++group) {
            auto & rows = grouped_slots[group];
            std::sort(rows.begin(), rows.end());
            const int rows_begin = static_cast<int>(grouped_rows_host.size());
            grouped_rows_host.insert(grouped_rows_host.end(), rows.begin(), rows.end());
            for (int row_start = 0; row_start < static_cast<int>(rows.size()); row_start += exec_n) {
                grouped_chunks_host.push_back(static_cast<int32_t>(group));
                grouped_row_starts_host.push_back(row_start);
            }
            grouped_offsets_host.push_back(rows_begin + static_cast<int>(rows.size()));
        }
        if (grouped_rows_host.size() != ids_count || grouped_chunks_host.empty()) {
            cleanup();
            error = "mxfp4_mmv_id_xmx_tiled grouped I8 setup failed.";
            return false;
        }
        d_grouped_experts    = sycl::malloc_device<int32_t>(grouped_experts_host.size(), queue);
        d_grouped_offsets    = sycl::malloc_device<int32_t>(grouped_offsets_host.size(), queue);
        d_grouped_rows       = sycl::malloc_device<int32_t>(grouped_rows_host.size(), queue);
        d_grouped_chunks     = sycl::malloc_device<int32_t>(grouped_chunks_host.size(), queue);
        d_grouped_row_starts = sycl::malloc_device<int32_t>(grouped_row_starts_host.size(), queue);
        if (!d_grouped_experts || !d_grouped_offsets || !d_grouped_rows || !d_grouped_chunks || !d_grouped_row_starts) {
            cleanup();
            error = "device allocation failed for grouped mxfp4_mmv_id_xmx_tiled.";
            return false;
        }
    }

    queue.memcpy(d_tiled, launch_slices.data(), launch_weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), expert_slots * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    if (i8_rowmajor) {
        queue.memcpy(d_grouped_experts, grouped_experts_host.data(), grouped_experts_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_offsets, grouped_offsets_host.data(), grouped_offsets_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_rows, grouped_rows_host.data(), grouped_rows_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_chunks, grouped_chunks_host.data(), grouped_chunks_host.size() * sizeof(int32_t));
        queue.memcpy(d_grouped_row_starts, grouped_row_starts_host.data(),
                     grouped_row_starts_host.size() * sizeof(int32_t));
    }
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
    args.dpas_b_packed      = d_dpas_b;
    args.dpas_y_scales      = d_dpas_y;
    args.grouped_expert_ids = d_grouped_experts;
    args.grouped_offsets    = d_grouped_offsets;
    args.grouped_row_slots  = d_grouped_rows;
    args.grouped_chunks     = d_grouped_chunks;
    args.grouped_row_starts = d_grouped_row_starts;
    args.ncols              = static_cast<int>(k);
    args.ncols_y            = static_cast<int>(k);
    args.nrows_per_expert   = static_cast<int>(m);
    args.num_experts        = static_cast<int>(expert_slots);
    args.n_ids              = static_cast<int>(n_selected);
    args.n_tokens           = static_cast<int>(n_tokens);
    args.ne11               = 1;
    args.xmx_tiles_n        = tiles_n;
    args.raw_accum          = raw_accum;
    args.i8_rowmajor        = i8_rowmajor;
    args.grouped_n_chunks   = static_cast<int>(grouped_chunks_host.size());
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
