#ifndef GGML_SYCL_DSV4_HC_KERNELS_HPP
#define GGML_SYCL_DSV4_HC_KERNELS_HPP

// DeepSeek-V4 style hyper-connection kernels: GGML_OP_DSV4_HC_PRE / _COMB / _POST.
//
// Header-only and free of every backend header on purpose: a launcher takes a plain sycl::queue and
// raw pointers, so tests/test-sycl-dsv4-hc-kernels can compile the SAME kernels and run them on the
// opencl:cpu device against an independent oracle, with no model, no GPU and no unified cache.
// dsv4-hc.cpp is the only production caller; it resolves the ggml tensors into these argument structs.
//
// The semantics are those of the CPU reference (ggml-cpu/ops.cpp ggml_compute_forward_dsv4_hc_*):
//   pre   dst[i, t]    = scale * sum_h x[i, h, t] * (gated ? sigmoid(w[i, h, t]) : w[h, t])
//   comb  dst[:, :, t] = sinkhorn(softmax(mixes[8 + 4*src + dst, t] * scale[2] + base[...]) + eps)
//   post  dst[i, d, t] = x[i, t] * post[d, t] + (comb ? sum_s residual[i, s, t] * comb[d, s, t]
//                                                     : residual[i, d, t])
// Every stride is in ELEMENTS (float), not bytes. Nothing is allocated and nothing blocks the host:
// each launcher submits one kernel on the caller's in-order queue, so it is recordable into a SYCL graph.

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

namespace ggml_sycl_dsv4 {

// The comb op is defined for four streams only (ggml_dsv4_hc_comb asserts hc == 4).
constexpr int HC_COMB_STREAMS = 4;

constexpr int64_t HC_BLOCK_SIZE = 256;

struct hc_pre_args {
    const float * x;    // [n_embd, hc, n_tokens]
    const float * w;    // gated: [n_embd, hc, n_tokens] gate logits; else [hc, n_tokens] weights
    float *       dst;  // [n_embd, n_tokens]
    int64_t       n_embd;
    int64_t       hc;
    int64_t       n_tokens;
    int64_t       sx0, sx1, sx2;
    int64_t       sw0, sw1, sw2;  // sw2 is unused when not gated
    int64_t       sd0, sd1;
    float         scale;
    bool          gated;
};

inline void hc_pre_launch(sycl::queue & q, const hc_pre_args & a) {
    const int64_t nr         = a.n_embd * a.n_tokens;
    const int64_t num_blocks = (nr + HC_BLOCK_SIZE - 1) / HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_blocks * HC_BLOCK_SIZE), sycl::range<1>(HC_BLOCK_SIZE)),
                   [=](sycl::nd_item<1> item) {
                       const int64_t ir = item.get_global_id(0);
                       if (ir >= nr) {
                           return;
                       }

                       const int64_t i0 = ir % a.n_embd;
                       const int64_t it = ir / a.n_embd;

                       float sum = 0.0f;
                       for (int64_t ih = 0; ih < a.hc; ++ih) {
                           const float xv = a.x[i0 * a.sx0 + ih * a.sx1 + it * a.sx2];
                           float       wv;
                           if (a.gated) {
                               const float gv = a.w[i0 * a.sw0 + ih * a.sw1 + it * a.sw2];
                               wv             = 1.0f / (1.0f + sycl::exp(-gv));
                           } else {
                               wv = a.w[ih * a.sw0 + it * a.sw1];
                           }
                           sum += xv * wv;
                       }

                       a.dst[i0 * a.sd0 + it * a.sd1] = a.scale * sum;
                   });
}

struct hc_comb_args {
    const float * mixes;  // [(2 + 4) * 4, n_tokens]
    const float * scale;  // [>= 3]; element 2 scales the comb mixes
    const float * base;   // [(2 + 4) * 4]
    float *       dst;    // [4, 4, n_tokens]
    int64_t       n_tokens;
    int64_t       sm0, sm1;
    int64_t       ss0;
    int64_t       sb0;
    int64_t       sd0, sd1, sd2;
    float         eps;
    int32_t       n_iter;
};

inline void hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < HC_COMB_STREAMS; ++isrc) {
            sum += comb[idst + HC_COMB_STREAMS * isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < HC_COMB_STREAMS; ++isrc) {
            comb[idst + HC_COMB_STREAMS * isrc] *= inv_sum;
        }
    }
}

inline void hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < HC_COMB_STREAMS; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
            sum += comb[idst + HC_COMB_STREAMS * isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
            comb[idst + HC_COMB_STREAMS * isrc] *= inv_sum;
        }
    }
}

inline void hc_comb_launch(sycl::queue & q, const hc_comb_args & a) {
    constexpr int comb_offset = 2 * HC_COMB_STREAMS;

    const int64_t num_blocks = (a.n_tokens + HC_BLOCK_SIZE - 1) / HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_blocks * HC_BLOCK_SIZE), sycl::range<1>(HC_BLOCK_SIZE)),
                   [=](sycl::nd_item<1> item) {
                       const int64_t it = item.get_global_id(0);
                       if (it >= a.n_tokens) {
                           return;
                       }

                       const float scale_comb = a.scale[2 * a.ss0];
                       float       comb[HC_COMB_STREAMS * HC_COMB_STREAMS];

                       for (int isrc = 0; isrc < HC_COMB_STREAMS; ++isrc) {
                           float max = -std::numeric_limits<float>::infinity();
                           for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
                               const int   idx = idst + HC_COMB_STREAMS * isrc;
                               const float v   = a.mixes[(comb_offset + idx) * a.sm0 + it * a.sm1] * scale_comb +
                                               a.base[(comb_offset + idx) * a.sb0];
                               comb[idx] = v;
                               max       = sycl::fmax(max, v);
                           }

                           float sum = 0.0f;
                           for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
                               const int   idx = idst + HC_COMB_STREAMS * isrc;
                               const float v   = sycl::exp(comb[idx] - max);
                               comb[idx]       = v;
                               sum += v;
                           }

                           const float inv_sum = 1.0f / sum;
                           for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
                               const int idx = idst + HC_COMB_STREAMS * isrc;
                               comb[idx]     = comb[idx] * inv_sum + a.eps;
                           }
                       }

                       hc_comb_norm_cols(comb, a.eps);
                       for (int32_t i = 1; i < a.n_iter; ++i) {
                           hc_comb_norm_rows(comb, a.eps);
                           hc_comb_norm_cols(comb, a.eps);
                       }

                       for (int isrc = 0; isrc < HC_COMB_STREAMS; ++isrc) {
                           for (int idst = 0; idst < HC_COMB_STREAMS; ++idst) {
                               const int idx                                   = idst + HC_COMB_STREAMS * isrc;
                               a.dst[idst * a.sd0 + isrc * a.sd1 + it * a.sd2] = comb[idx];
                           }
                       }
                   });
}

struct hc_post_args {
    const float * x;         // [n_embd, n_tokens]
    const float * residual;  // [n_embd, hc, n_tokens]
    const float * post;      // [hc, n_tokens]
    const float * comb;      // [hc, hc, n_tokens], or nullptr for identity mixing
    float *       dst;       // [n_embd, hc, n_tokens]
    int64_t       n_embd;
    int64_t       hc;
    int64_t       n_tokens;
    int64_t       sx0, sx1;
    int64_t       sr0, sr1, sr2;
    int64_t       sp0, sp1;
    int64_t       sc0, sc1, sc2;  // unused when comb is nullptr
    int64_t       sd0, sd1, sd2;
};

inline void hc_post_launch(sycl::queue & q, const hc_post_args & a) {
    const int64_t nr         = a.n_embd * a.hc * a.n_tokens;
    const int64_t num_blocks = (nr + HC_BLOCK_SIZE - 1) / HC_BLOCK_SIZE;

    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_blocks * HC_BLOCK_SIZE), sycl::range<1>(HC_BLOCK_SIZE)),
                   [=](sycl::nd_item<1> item) {
                       const int64_t ir = item.get_global_id(0);
                       if (ir >= nr) {
                           return;
                       }

                       const int64_t i0   = ir % a.n_embd;
                       const int64_t idst = (ir / a.n_embd) % a.hc;
                       const int64_t it   = ir / (a.n_embd * a.hc);

                       float sum = a.x[i0 * a.sx0 + it * a.sx1] * a.post[idst * a.sp0 + it * a.sp1];
                       if (a.comb != nullptr) {
                           for (int64_t isrc = 0; isrc < a.hc; ++isrc) {
                               sum += a.residual[i0 * a.sr0 + isrc * a.sr1 + it * a.sr2] *
                                      a.comb[idst * a.sc0 + isrc * a.sc1 + it * a.sc2];
                           }
                       } else {
                           sum += a.residual[i0 * a.sr0 + idst * a.sr1 + it * a.sr2];
                       }

                       a.dst[i0 * a.sd0 + idst * a.sd1 + it * a.sd2] = sum;
                   });
}

}  // namespace ggml_sycl_dsv4

#endif  // GGML_SYCL_DSV4_HC_KERNELS_HPP
