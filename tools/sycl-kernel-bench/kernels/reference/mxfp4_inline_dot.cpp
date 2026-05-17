#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "reference_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

bool run_mxfp4_pair_glu(const GeneratedWeights &     weights,
                        const GeneratedActivations & activations,
                        int64_t                      m,
                        int64_t                      n_selected,
                        int64_t                      k,
                        int64_t                      n_tokens,
                        int                          rows_per_wg,
                        bool                         cache_y,
                        bool                         direct_xmx,
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
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_pair_glu requires SOA MXFP4 weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_pair_glu requires SOA Q8_1 activations.";
        return false;
    }

    const size_t  selected_count = static_cast<size_t>(n_selected);
    const size_t  token_count    = static_cast<size_t>(n_tokens);
    const size_t  ids_count      = selected_count * token_count;
    const size_t  expert_bytes   = weights.layout.size();
    const size_t  weight_bytes   = expert_bytes * selected_count;
    const size_t  act_bytes      = activations.q8_1.size();
    const size_t  out_count      = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes      = out_count * sizeof(float);
    const int64_t q8_row_bytes   = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);

    std::vector<uint8_t> gate_slices(weight_bytes);
    std::vector<uint8_t> up_slices(weight_bytes);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        std::copy(weights.layout.begin(), weights.layout.end(), gate_slices.begin() + sel * expert_bytes);
        std::copy(weights.layout.begin(), weights.layout.end(), up_slices.begin() + sel * expert_bytes);
    }

    uint8_t *        d_gate      = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    uint8_t *        d_up        = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_gate_ptrs = sycl::malloc_device<const uint8_t *>(selected_count, queue);
    const uint8_t ** d_up_ptrs   = sycl::malloc_device<const uint8_t *>(selected_count, queue);
    uint8_t *        d_act       = sycl::malloc_device<uint8_t>(act_bytes, queue);
    int32_t *        d_ids       = sycl::malloc_device<int32_t>(ids_count, queue);
    float *          d_out       = sycl::malloc_device<float>(out_count, queue);
    float *          d_ref_out   = direct_xmx && validate ? sycl::malloc_device<float>(out_count, queue) : nullptr;

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
        if (d_ref_out) {
            sycl::free(d_ref_out, queue);
        }
    };

    if (!d_gate || !d_up || !d_gate_ptrs || !d_up_ptrs || !d_act || !d_ids || !d_out ||
        (direct_xmx && validate && !d_ref_out)) {
        cleanup();
        error = "device allocation failed for mxfp4_pair_glu.";
        return false;
    }

    std::vector<const uint8_t *> host_gate_ptrs(selected_count);
    std::vector<const uint8_t *> host_up_ptrs(selected_count);
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        host_gate_ptrs[sel] = d_gate + sel * expert_bytes;
        host_up_ptrs[sel]   = d_up + sel * expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            host_ids[token * selected_count + sel] = static_cast<int32_t>(sel);
        }
    }

    queue.memcpy(d_gate, gate_slices.data(), weight_bytes);
    queue.memcpy(d_up, up_slices.data(), weight_bytes);
    queue.memcpy(d_gate_ptrs, host_gate_ptrs.data(), selected_count * sizeof(const uint8_t *));
    queue.memcpy(d_up_ptrs, host_up_ptrs.data(), selected_count * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    ggml_sycl::mxfp4_pair_glu_bench_args args{};
    args.stream             = &queue;
    args.gate_ptrs          = reinterpret_cast<const void * const *>(d_gate_ptrs);
    args.up_ptrs            = reinterpret_cast<const void * const *>(d_up_ptrs);
    args.activations_q8_soa = d_act;
    args.output             = d_out;
    args.ids                = d_ids;
    args.ncols              = static_cast<int>(k);
    args.ncols_y            = static_cast<int>(k);
    args.nrows_per_expert   = static_cast<int>(m);
    args.n_ids              = static_cast<int>(n_selected);
    args.n_tokens           = static_cast<int>(n_tokens);
    args.ne11               = 1;
    args.ids_nb0            = sizeof(int32_t);
    args.ids_nb1            = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.nb11               = q8_row_bytes;
    args.nb12               = q8_row_bytes;
    args.dst_nb1            = static_cast<int64_t>(m * sizeof(float));
    args.dst_nb2            = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.rows_per_wg        = rows_per_wg;
    args.cache_y            = cache_y;
    args.direct_xmx         = direct_xmx;
    args.glu_op             = GGML_GLU_OP_SWIGLU_OAI;
    args.alpha              = 1.702f;
    args.limit              = 7.0f;

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
    const double bytes = 2.0 * static_cast<double>(weight_bytes) * static_cast<double>(n_tokens) +
                         static_cast<double>(act_bytes) + static_cast<double>(out_bytes) +
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
        if (direct_xmx) {
            queue.memset(d_ref_out, 0, out_bytes).wait();
            ggml_sycl::mxfp4_pair_glu_bench_args ref_args = args;
            ref_args.output                               = d_ref_out;
            ref_args.direct_xmx                           = false;
            ref_args.rows_per_wg                          = 1;
            ref_args.cache_y                              = false;
            if (!ggml_sycl::ggml_sycl_mxfp4_pair_glu_bench_launch(ref_args)) {
                cleanup();
                error = "mxfp4_pair_glu direct-XMX reference launch rejected.";
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
                                  "mxfp4_pair_glu direct-XMX validation failed at %zu: actual=%.6f expected=%.6f "
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

bool run_mxfp4_mmv_id(const GeneratedWeights &     weights,
                      const GeneratedActivations & activations,
                      int64_t                      m,
                      int64_t                      n_selected,
                      int64_t                      k,
                      int64_t                      n_tokens,
                      int                          rows_per_wg,
                      bool                         validate,
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
    if (weights.layout_mode != GGML_LAYOUT_SOA || weights.layout.empty()) {
        error = "mxfp4_mmv_id requires SOA MXFP4 weights.";
        return false;
    }
    if (activations.q8_1.empty()) {
        error = "mxfp4_mmv_id requires SOA Q8_1 activations.";
        return false;
    }

    const size_t  selected_count = static_cast<size_t>(n_selected);
    const size_t  token_count    = static_cast<size_t>(n_tokens);
    const size_t  ids_count      = selected_count * token_count;
    const size_t  expert_bytes   = weights.layout.size();
    const size_t  weight_bytes   = expert_bytes * selected_count;
    const size_t  act_bytes      = activations.q8_1.size();
    const size_t  out_count      = static_cast<size_t>(m) * selected_count * token_count;
    const size_t  out_bytes      = out_count * sizeof(float);
    const int64_t q8_row_bytes   = static_cast<int64_t>(k * sizeof(block_q8_1) / QK8_1);

    std::vector<uint8_t> weight_slices(weight_bytes);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        std::copy(weights.layout.begin(), weights.layout.end(), weight_slices.begin() + sel * expert_bytes);
    }

    uint8_t *        d_weights = sycl::malloc_device<uint8_t>(weight_bytes, queue);
    const uint8_t ** d_ptrs    = sycl::malloc_device<const uint8_t *>(selected_count, queue);
    uint8_t *        d_act     = sycl::malloc_device<uint8_t>(act_bytes, queue);
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
        error = "device allocation failed for mxfp4_mmv_id.";
        return false;
    }

    std::vector<const uint8_t *> host_ptrs(selected_count);
    std::vector<int32_t>         host_ids(ids_count);
    for (size_t sel = 0; sel < selected_count; ++sel) {
        host_ptrs[sel] = d_weights + sel * expert_bytes;
    }
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t sel = 0; sel < selected_count; ++sel) {
            host_ids[token * selected_count + sel] = static_cast<int32_t>(sel);
        }
    }

    queue.memcpy(d_weights, weight_slices.data(), weight_bytes);
    queue.memcpy(d_ptrs, host_ptrs.data(), selected_count * sizeof(const uint8_t *));
    queue.memcpy(d_act, activations.q8_1.data(), act_bytes);
    queue.memcpy(d_ids, host_ids.data(), ids_count * sizeof(int32_t));
    queue.memset(d_out, 0, out_bytes);
    queue.wait_and_throw();

    ggml_sycl::mxfp4_mmv_id_bench_args args{};
    args.stream             = &queue;
    args.expert_ptrs        = reinterpret_cast<const void * const *>(d_ptrs);
    args.activations_q8_soa = d_act;
    args.output             = d_out;
    args.ids                = d_ids;
    args.ncols              = static_cast<int>(k);
    args.ncols_y            = static_cast<int>(k);
    args.nrows_per_expert   = static_cast<int>(m);
    args.num_experts        = static_cast<int>(n_selected);
    args.n_ids              = static_cast<int>(n_selected);
    args.n_tokens           = static_cast<int>(n_tokens);
    args.ne11               = 1;
    args.ids_nb0            = sizeof(int32_t);
    args.ids_nb1            = static_cast<int64_t>(selected_count * sizeof(int32_t));
    args.nb11               = q8_row_bytes;
    args.nb12               = q8_row_bytes;
    args.dst_nb1            = static_cast<int64_t>(m * sizeof(float));
    args.dst_nb2            = static_cast<int64_t>(selected_count * static_cast<size_t>(m) * sizeof(float));
    args.rows_per_wg        = rows_per_wg;

    auto launch = [&]() {
        if (!ggml_sycl::ggml_sycl_mxfp4_mmv_id_bench_launch(args)) {
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
    const double bytes = static_cast<double>(weight_bytes) * static_cast<double>(n_tokens) +
                         static_cast<double>(act_bytes) + static_cast<double>(out_bytes) +
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
    }

    cleanup();
    return true;
}

}  // namespace sycl_bench
