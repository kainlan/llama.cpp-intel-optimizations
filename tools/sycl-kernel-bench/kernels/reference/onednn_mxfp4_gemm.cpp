#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "reference_kernels.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <sstream>
#include <unordered_map>

namespace sycl_bench {

#if !GGML_SYCL_DNNL
bool run_onednn_mxfp4_gemm(const GeneratedWeights &,
                           const GeneratedActivations &,
                           int64_t,
                           int64_t,
                           int64_t,
                           int,
                           int,
                           bool,
                           int,
                           sycl::queue &,
                           ReferenceMetrics &,
                           std::string & error) {
    error = "oneDNN not enabled in this build (GGML_SYCL_DNNL=0)";
    return false;
}
#else

static void pack_mxfp4_to_onednn_km(const GeneratedWeights & weights,
                                    int64_t                  m,
                                    int64_t                  k,
                                    std::vector<uint8_t> &   out_f4,
                                    std::vector<uint8_t> &   out_e8m0,
                                    std::vector<float> &     out_f32) {
    const int64_t groups = k / QK_MXFP4;
    out_f4.assign((static_cast<size_t>(k) * static_cast<size_t>(m) + 1u) / 2u, 0);
    out_e8m0.assign(static_cast<size_t>(m) * static_cast<size_t>(groups), 0);
    out_f32.assign(static_cast<size_t>(m) * static_cast<size_t>(groups), 0.0f);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_MXFP4, k);
    for (int64_t row = 0; row < m; ++row) {
        const uint8_t *     row_ptr = weights.aos.data() + static_cast<size_t>(row) * row_bytes;
        const block_mxfp4 * blocks  = reinterpret_cast<const block_mxfp4 *>(row_ptr);
        for (int64_t group = 0; group < groups; ++group) {
            const block_mxfp4 & block = blocks[group];
            const size_t scale_idx    = static_cast<size_t>(group) * static_cast<size_t>(m) + static_cast<size_t>(row);
            out_e8m0[scale_idx]       = block.e;
            out_f32[scale_idx]        = GGML_E8M0_TO_FP32(block.e);

            for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                const uint8_t packed = block.qs[i];
                const uint8_t lo     = packed & 0x0f;
                const uint8_t hi     = packed >> 4;

                const int64_t k0   = group * QK_MXFP4 + i;
                const int64_t k1   = k0 + QK_MXFP4 / 2;
                const size_t  idx0 = static_cast<size_t>(k0) * static_cast<size_t>(m) + static_cast<size_t>(row);
                const size_t  idx1 = static_cast<size_t>(k1) * static_cast<size_t>(m) + static_cast<size_t>(row);

                if ((idx0 & 1u) == 0) {
                    out_f4[idx0 / 2u] = static_cast<uint8_t>((out_f4[idx0 / 2u] & 0xf0) | lo);
                } else {
                    out_f4[idx0 / 2u] = static_cast<uint8_t>((out_f4[idx0 / 2u] & 0x0f) | (lo << 4));
                }
                if ((idx1 & 1u) == 0) {
                    out_f4[idx1 / 2u] = static_cast<uint8_t>((out_f4[idx1 / 2u] & 0xf0) | hi);
                } else {
                    out_f4[idx1 / 2u] = static_cast<uint8_t>((out_f4[idx1 / 2u] & 0x0f) | (hi << 4));
                }
            }
        }
    }
}

static bool validate_mxfp4_result(const GeneratedWeights &       weights,
                                  const GeneratedActivations &   activations,
                                  const std::vector<ggml_half> & actual,
                                  int64_t                        m,
                                  int64_t                        n,
                                  int64_t                        k,
                                  std::string &                  error) {
    std::vector<float> weight_row(static_cast<size_t>(k));
    std::vector<float> ref(static_cast<size_t>(n) * static_cast<size_t>(m), 0.0f);
    std::string        dequant_error;
    const size_t       row_bytes = ggml_row_size(GGML_TYPE_MXFP4, k);

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
            ref[static_cast<size_t>(b) * static_cast<size_t>(m) + static_cast<size_t>(row)] = sum;
        }
    }

    double max_abs       = 0.0;
    double mean_abs      = 0.0;
    size_t nonfinite_got = 0;
    size_t nonfinite_ref = 0;
    size_t first_bad     = ref.size();
    for (size_t i = 0; i < ref.size(); ++i) {
        const double got = static_cast<double>(bench_half_to_float(actual[i]));
        if (!std::isfinite(got)) {
            ++nonfinite_got;
            if (first_bad == ref.size()) {
                first_bad = i;
            }
        }
        if (!std::isfinite(ref[i])) {
            ++nonfinite_ref;
            if (first_bad == ref.size()) {
                first_bad = i;
            }
        }
        const double diff = std::fabs(got - static_cast<double>(ref[i]));
        max_abs           = std::max(max_abs, diff);
        mean_abs += diff;
    }
    if (!ref.empty()) {
        mean_abs /= static_cast<double>(ref.size());
    }

    if (max_abs > 0.25 && mean_abs > 0.02) {
        std::ostringstream oss;
        oss << "oneDNN MXFP4 validation failed: max_abs=" << max_abs << " mean_abs=" << mean_abs;
        if (nonfinite_got || nonfinite_ref) {
            oss << " nonfinite_got=" << nonfinite_got << " nonfinite_ref=" << nonfinite_ref;
        }
        if (first_bad < ref.size()) {
            oss << " first_bad=" << first_bad << " got=" << bench_half_to_float(actual[first_bad])
                << " ref=" << ref[first_bad];
        }
        error = oss.str();
        return false;
    }
    return true;
}

bool run_onednn_mxfp4_gemm(const GeneratedWeights &     weights,
                           const GeneratedActivations & activations,
                           int64_t                      m,
                           int64_t                      n,
                           int64_t                      k,
                           int                          warmup,
                           int                          iterations,
                           bool                         validate,
                           int                          scale_mode,
                           sycl::queue &                queue,
                           ReferenceMetrics &           out,
                           std::string &                error) {
    if (m <= 0 || n <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "onednn_mxfp4_gemm requires positive dims and K divisible by QK_MXFP4.";
        return false;
    }
    if (activations.fp16.empty()) {
        error = "FP16 activations missing for onednn_mxfp4_gemm.";
        return false;
    }

    try {
        std::vector<uint8_t> weights_f4;
        std::vector<uint8_t> scales_e8m0;
        std::vector<float>   scales_f32;

        auto t0 = std::chrono::high_resolution_clock::now();
        pack_mxfp4_to_onednn_km(weights, m, k, weights_f4, scales_e8m0, scales_f32);

        auto eng    = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
        auto stream = dnnl::sycl_interop::make_stream(eng, queue);

        const size_t src_bytes      = activations.fp16.size() * sizeof(ggml_half);
        const size_t wei_user_bytes = weights_f4.size();
        const size_t dst_bytes      = static_cast<size_t>(n) * static_cast<size_t>(m) * sizeof(ggml_half);
        const bool   use_f32_scales = scale_mode != 0;
        const size_t scale_bytes =
            use_f32_scales ? scales_f32.size() * sizeof(float) : scales_e8m0.size() * sizeof(uint8_t);

        ggml_half * src_dev      = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, src_bytes, queue));
        uint8_t *   wei_user_dev = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, wei_user_bytes, queue));
        uint8_t *   scales_dev   = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, scale_bytes, queue));
        ggml_half * dst_dev      = static_cast<ggml_half *>(sycl::aligned_alloc_device(64, dst_bytes, queue));

        if (!src_dev || !wei_user_dev || !scales_dev || !dst_dev) {
            if (src_dev) {
                sycl::free(src_dev, queue);
            }
            if (wei_user_dev) {
                sycl::free(wei_user_dev, queue);
            }
            if (scales_dev) {
                sycl::free(scales_dev, queue);
            }
            if (dst_dev) {
                sycl::free(dst_dev, queue);
            }
            error = "device allocation failed for onednn_mxfp4_gemm.";
            return false;
        }

        queue.memcpy(src_dev, activations.fp16.data(), src_bytes);
        queue.memcpy(wei_user_dev, weights_f4.data(), wei_user_bytes);
        if (use_f32_scales) {
            queue.memcpy(scales_dev, scales_f32.data(), scale_bytes);
        } else {
            queue.memcpy(scales_dev, scales_e8m0.data(), scale_bytes);
        }
        queue.wait_and_throw();

        dnnl::memory::desc src_md({ n, k }, dnnl::memory::data_type::f16, { k, 1 });
        dnnl::memory::desc wei_user_md({ k, m }, dnnl::memory::data_type::f4_e2m1, { m, 1 });
        dnnl::memory::desc wei_any_md({ k, m }, dnnl::memory::data_type::f4_e2m1, dnnl::memory::format_tag::any);
        dnnl::memory::desc dst_md({ n, m }, dnnl::memory::data_type::f16, { m, 1 });

        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { QK_MXFP4, 1 },
                        use_f32_scales ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::e8m0);
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16);
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

        auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_md, wei_any_md, dst_md, attr);

        dnnl::memory src_mem(src_md, eng, src_dev);
        dnnl::memory wei_user_mem(wei_user_md, eng, wei_user_dev);
        dnnl::memory dst_mem(dst_md, eng, dst_dev);
        dnnl::memory scales_mem(
            {
                { m, k / QK_MXFP4 },
                use_f32_scales ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::e8m0,
                { 1, m            }
        },
            eng, scales_dev);

        dnnl::memory wei_mem        = wei_user_mem;
        uint8_t *    wei_packed_dev = nullptr;
        if (matmul_pd.weights_desc() != wei_user_mem.get_desc()) {
            const size_t wei_packed_bytes = matmul_pd.weights_desc().get_size();
            wei_packed_dev = static_cast<uint8_t *>(sycl::aligned_alloc_device(64, wei_packed_bytes, queue));
            if (!wei_packed_dev) {
                sycl::free(src_dev, queue);
                sycl::free(wei_user_dev, queue);
                sycl::free(scales_dev, queue);
                sycl::free(dst_dev, queue);
                error = "device allocation failed for oneDNN packed MXFP4 weights.";
                return false;
            }
            wei_mem = dnnl::memory(matmul_pd.weights_desc(), eng, wei_packed_dev);
            dnnl::reorder(wei_user_mem, wei_mem).execute(stream, wei_user_mem, wei_mem);
            stream.wait();
        }

        auto t1        = std::chrono::high_resolution_clock::now();
        out.dequant_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        auto                                  matmul_prim = dnnl::matmul(matmul_pd);
        std::unordered_map<int, dnnl::memory> args;
        args.insert({ DNNL_ARG_SRC, src_mem });
        args.insert({ DNNL_ARG_WEIGHTS, wei_mem });
        args.insert({ DNNL_ARG_DST, dst_mem });
        args.insert({ DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scales_mem });

        for (int i = 0; i < warmup; ++i) {
            matmul_prim.execute(stream, args);
        }
        stream.wait();

        auto g0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            matmul_prim.execute(stream, args);
        }
        stream.wait();
        auto g1 = std::chrono::high_resolution_clock::now();

        const double gemm_total_us = std::chrono::duration<double, std::micro>(g1 - g0).count();
        out.gemm_us                = (iterations > 0) ? gemm_total_us / iterations : 0.0;
        out.total_us               = out.dequant_us + out.gemm_us;

        const double ops          = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        const double gemm_s       = out.gemm_us * 1e-6;
        out.tflops                = (gemm_s > 0.0) ? (ops / gemm_s) / 1.0e12 : 0.0;
        const double steady_bytes = static_cast<double>(wei_user_bytes + src_bytes + dst_bytes + scale_bytes);
        out.bandwidth_gbps        = (gemm_s > 0.0) ? (steady_bytes / gemm_s) / 1.0e9 : 0.0;

        if (validate) {
            std::vector<ggml_half> actual(static_cast<size_t>(n) * static_cast<size_t>(m));
            queue.memcpy(actual.data(), dst_dev, dst_bytes).wait();
            if (!validate_mxfp4_result(weights, activations, actual, m, n, k, error)) {
                sycl::free(src_dev, queue);
                sycl::free(wei_user_dev, queue);
                sycl::free(scales_dev, queue);
                sycl::free(dst_dev, queue);
                if (wei_packed_dev) {
                    sycl::free(wei_packed_dev, queue);
                }
                return false;
            }
        }

        sycl::free(src_dev, queue);
        sycl::free(wei_user_dev, queue);
        sycl::free(scales_dev, queue);
        sycl::free(dst_dev, queue);
        if (wei_packed_dev) {
            sycl::free(wei_packed_dev, queue);
        }
        return true;
    } catch (const std::exception & e) {
        error = std::string("oneDNN MXFP4 GEMM failed: ") + e.what();
        return false;
    }
}
#endif

}  // namespace sycl_bench
