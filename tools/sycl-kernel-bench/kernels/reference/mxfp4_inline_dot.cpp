#include "reference_kernels.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <type_traits>

namespace sycl_bench {

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

}  // namespace sycl_bench
