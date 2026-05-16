#include "reference_kernels.hpp"

#include <chrono>
#include <cstdint>

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

bool run_mxfp4_decode_bandwidth(const GeneratedWeights & weights,
                                int64_t                  m,
                                int64_t                  k,
                                ggml_layout_mode         layout,
                                int                      warmup,
                                int                      iterations,
                                sycl::queue &            queue,
                                ReferenceMetrics &       out,
                                std::string &            error) {
    if (m <= 0 || k <= 0 || (k % QK_MXFP4) != 0) {
        error = "mxfp4_decode requires positive M and K divisible by QK_MXFP4";
        return false;
    }
    if (layout != GGML_LAYOUT_AOS && layout != GGML_LAYOUT_SOA) {
        error = "mxfp4_decode supports AOS and SOA layouts only";
        return false;
    }

    const std::vector<uint8_t> & host_weights = (layout == GGML_LAYOUT_AOS) ? weights.aos : weights.layout;
    if (host_weights.empty()) {
        error = "mxfp4_decode received empty weight buffer";
        return false;
    }

    const size_t  input_bytes    = host_weights.size();
    const size_t  output_count   = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t  output_bytes   = output_count * sizeof(float);
    const int64_t blocks_per_row = k / QK_MXFP4;
    const size_t  nblocks        = static_cast<size_t>(m) * static_cast<size_t>(blocks_per_row);

    uint8_t * d_weights = sycl::malloc_device<uint8_t>(input_bytes, queue);
    float *   d_output  = sycl::malloc_device<float>(output_count, queue);
    if (!d_weights || !d_output) {
        if (d_weights) {
            sycl::free(d_weights, queue);
        }
        if (d_output) {
            sycl::free(d_output, queue);
        }
        error = "device allocation failed for mxfp4_decode";
        return false;
    }

    queue.memcpy(d_weights, host_weights.data(), input_bytes).wait();

    auto submit_decode = [&]() {
        return queue.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(nblocks), [=](sycl::id<1> idx) {
                const size_t    block_idx    = idx[0];
                const size_t    row          = block_idx / static_cast<size_t>(blocks_per_row);
                const size_t    block_in_row = block_idx - row * static_cast<size_t>(blocks_per_row);
                uint8_t         e            = 0;
                const uint8_t * qs           = nullptr;

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

                const float scale   = mxfp4_e8m0_to_fp32_device(e);
                float *     out_row = d_output + row * static_cast<size_t>(k) + block_in_row * QK_MXFP4;
                for (int i = 0; i < QK_MXFP4 / 2; ++i) {
                    const uint8_t packed      = qs[i];
                    out_row[i]                = scale * mxfp4_value_device(packed & 0x0f);
                    out_row[i + QK_MXFP4 / 2] = scale * mxfp4_value_device(packed >> 4);
                }
            });
        });
    };

    for (int i = 0; i < warmup; ++i) {
        submit_decode();
    }
    queue.wait_and_throw();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit_decode();
    }
    queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us    = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double mean_us     = (iterations > 0) ? total_us / iterations : 0.0;
    const double mean_s      = mean_us * 1e-6;
    const double bytes_moved = static_cast<double>(input_bytes + output_bytes);

    out.total_us       = mean_us;
    out.dequant_us     = mean_us;
    out.bandwidth_gbps = (mean_s > 0.0) ? (bytes_moved / mean_s) / 1e9 : 0.0;

    sycl::free(d_weights, queue);
    sycl::free(d_output, queue);
    return true;
}

}  // namespace sycl_bench
