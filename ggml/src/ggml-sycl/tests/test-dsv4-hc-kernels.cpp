// DSV4_HC_PRE / _COMB / _POST and LIGHTNING_INDEXER kernels against an independent oracle (llama.cpp-pmzl).
//
// Host-only: the kernels run on the opencl:cpu SYCL device. No GPU, no model, no unified cache, no ggml graph.
// It compiles the SAME header-only kernels dsv4-hc.cpp / lightning-indexer.cpp launch
// (dsv4-hc-kernels.hpp, lightning-indexer-kernel.hpp), so the code under test is the shipped code.
//
// The oracle is written here, in double precision, from the op definitions in ggml.h -- it does not call
// the kernels, the backend, or the CPU backend's implementation. K for the quantized indexer cases is
// quantized with ggml-base's reference quantizers and the oracle decodes it with ggml-base's to_float, a
// third implementation of the formats that shares nothing with lightning_indexer_k_elem.
//
// Every check that matters has a control: `controls` runs the SAME kernel output against an oracle that
// is wrong in exactly the way a plausible kernel bug would be (sigmoid dropped, null comb mistaken for a
// zero comb, one Sinkhorn iteration short, ReLU dropped, mask dropped) and requires the comparison to
// REJECT it. A comparison that cannot reject those is decoration.
//
// Run: ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build/bin/test-sycl-dsv4-hc-kernels
// Exit 77 (ctest SKIP) when no opencl:cpu device is visible -- a skip is not a pass.

#include "../dsv4-hc-kernels.hpp"
#include "../lightning-indexer-kernel.hpp"
#include "sycl-test-skip.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace ggml_sycl_dsv4;

namespace {

int g_cases  = 0;
int g_failed = 0;

constexpr double NMSE_TOL     = 1e-9;
constexpr double NMSE_REJECTS = 1e-4;  // a control oracle must be at least this far from the kernel

std::mt19937 g_rng(20261003);

float rnd(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(g_rng);
}

// ---- USM buffers (test-only; the backend itself allocates nothing) ---------------------------------------------

struct buf {
    sycl::queue * q = nullptr;
    float *       p = nullptr;
    size_t        n = 0;

    buf(sycl::queue & q, size_t n) : q(&q), p(sycl::malloc_shared<float>(n ? n : 1, q)), n(n) {}

    buf(const buf &)             = delete;
    buf & operator=(const buf &) = delete;

    ~buf() { sycl::free(p, *q); }

    void fill(float lo, float hi) {
        for (size_t i = 0; i < n; ++i) {
            p[i] = rnd(lo, hi);
        }
    }

    void fill_value(float v) {
        for (size_t i = 0; i < n; ++i) {
            p[i] = v;
        }
    }
};

// A logical [n0, n1, n2] float view with element strides, and the span of storage it needs.
struct view3 {
    int64_t n0, n1, n2;
    int64_t s0, s1, s2;

    size_t span() const { return (size_t) (1 + (n0 - 1) * s0 + (n1 - 1) * s1 + (n2 - 1) * s2); }

    size_t at(int64_t i0, int64_t i1, int64_t i2) const { return (size_t) (i0 * s0 + i1 * s1 + i2 * s2); }
};

view3 contiguous(int64_t n0, int64_t n1, int64_t n2) {
    return { n0, n1, n2, 1, n0, n0 * n1 };
}

// Same logical shape, rows and planes padded: the strides a view of a larger tensor has.
view3 padded(int64_t n0, int64_t n1, int64_t n2) {
    const int64_t s1 = n0 + 3;
    return { n0, n1, n2, 1, s1, s1 * n1 + 5 };
}

// ---- comparison ------------------------------------------------------------------------------------------------

// NMSE of `got` against `want`; -inf scores must match exactly. Returns 1e30 on a non-finite mismatch.
double nmse(const std::vector<double> & got, const std::vector<double> & want) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (std::isinf(want[i]) || std::isinf(got[i])) {
            if (want[i] != got[i]) {
                return 1e30;
            }
            continue;
        }
        if (std::isnan(got[i]) || std::isnan(want[i])) {
            return 1e30;
        }
        num += (got[i] - want[i]) * (got[i] - want[i]);
        den += want[i] * want[i];
    }
    return den > 0.0 ? num / den : num;
}

void report(const std::string & name, bool ok, double err) {
    ++g_cases;
    if (!ok) {
        ++g_failed;
    }
    printf("%s: %s (nmse %.3e)\n", ok ? "PASS" : "FAIL", name.c_str(), err);
}

void expect_match(const std::string & name, const std::vector<double> & got, const std::vector<double> & want) {
    const double e = nmse(got, want);
    report(name, e <= NMSE_TOL, e);
}

void expect_reject(const std::string & name, const std::vector<double> & got, const std::vector<double> & wrong) {
    const double e = nmse(got, wrong);
    report("control: " + name + " is rejected", e >= NMSE_REJECTS, e);
}

// Storage outside the logical view must be exactly what the buffer held before the launch.
bool padding_untouched(const buf & b, const view3 & v, float sentinel) {
    std::vector<char> logical(b.n, 0);
    for (int64_t i2 = 0; i2 < v.n2; ++i2) {
        for (int64_t i1 = 0; i1 < v.n1; ++i1) {
            for (int64_t i0 = 0; i0 < v.n0; ++i0) {
                logical[v.at(i0, i1, i2)] = 1;
            }
        }
    }
    for (size_t i = 0; i < b.n; ++i) {
        if (!logical[i] && b.p[i] != sentinel) {
            return false;
        }
    }
    return true;
}

// ---- DSV4_HC_PRE -----------------------------------------------------------------------------------------------

std::vector<double> pre_oracle(const buf &   x,
                               const view3 & vx,
                               const buf &   w,
                               const view3 & vw,
                               bool          gated,
                               float         scale,
                               bool          wrong_no_sigmoid = false) {
    std::vector<double> out;
    for (int64_t t = 0; t < vx.n2; ++t) {
        for (int64_t i = 0; i < vx.n0; ++i) {
            double sum = 0.0;
            for (int64_t h = 0; h < vx.n1; ++h) {
                const double xv = x.p[vx.at(i, h, t)];
                double       wv;
                if (gated) {
                    const double g = w.p[vw.at(i, h, t)];
                    wv             = wrong_no_sigmoid ? g : 1.0 / (1.0 + std::exp(-g));
                } else {
                    wv = w.p[vw.at(h, t, 0)];
                }
                sum += xv * wv;
            }
            out.push_back((double) scale * sum);
        }
    }
    return out;
}

void test_pre(sycl::queue & q, int64_t n_embd, int64_t hc, int64_t n_tokens, bool gated, bool pad) {
    const view3 vx = pad ? padded(n_embd, hc, n_tokens) : contiguous(n_embd, hc, n_tokens);
    // gated: the weights are gate logits shaped like x; else [hc, n_tokens] per-stream weights
    const view3 vw = gated ? (pad ? padded(n_embd, hc, n_tokens) : contiguous(n_embd, hc, n_tokens)) :
                             (pad ? view3{ hc, n_tokens, 1, 1, hc + 2, (hc + 2) * n_tokens } :
                                    view3{ hc, n_tokens, 1, 1, hc, hc * n_tokens });
    const view3 vd = pad ? view3{ n_embd, n_tokens, 1, 1, n_embd + 7, (n_embd + 7) * n_tokens } :
                           view3{ n_embd, n_tokens, 1, 1, n_embd, n_embd * n_tokens };

    buf x(q, vx.span());
    buf w(q, vw.span());
    buf d(q, vd.span());
    x.fill(-1.0f, 1.0f);
    w.fill(gated ? -4.0f : 0.0f, gated ? 4.0f : 1.0f);
    d.fill_value(12345.0f);

    const float scale = gated ? 1.0f / (float) hc : 1.0f;

    hc_pre_args a = {};
    a.x           = x.p;
    a.w           = w.p;
    a.dst         = d.p;
    a.n_embd      = n_embd;
    a.hc          = hc;
    a.n_tokens    = n_tokens;
    a.sx0 = vx.s0, a.sx1 = vx.s1, a.sx2 = vx.s2;
    a.sw0 = vw.s0, a.sw1 = vw.s1, a.sw2 = gated ? vw.s2 : 0;
    a.sd0 = vd.s0, a.sd1 = vd.s1;
    a.scale = scale;
    a.gated = gated;
    hc_pre_launch(q, a);
    q.wait_and_throw();

    std::vector<double> got;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_embd; ++i) {
            got.push_back(d.p[(size_t) (i * vd.s0 + t * vd.s1)]);
        }
    }
    const std::string name = "pre n_embd=" + std::to_string(n_embd) + " hc=" + std::to_string(hc) +
                             " tokens=" + std::to_string(n_tokens) + (gated ? " gated" : " plain") +
                             (pad ? " strided" : "");
    expect_match(name, got, pre_oracle(x, vx, w, vw, gated, scale));
    if (pad) {
        report(name + " leaves padding alone", padding_untouched(d, vd, 12345.0f), 0.0);
    }
    if (gated && n_embd * n_tokens >= 64) {
        expect_reject("pre gated vs no sigmoid", got, pre_oracle(x, vx, w, vw, true, scale, true));
    }
}

// ---- DSV4_HC_COMB ----------------------------------------------------------------------------------------------

std::vector<double> comb_oracle(const buf & mixes,
                                int64_t     n_tokens,
                                const buf & scale,
                                const buf & base,
                                double      eps,
                                int         n_iter) {
    std::vector<double> out(16 * (size_t) n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        double m[4][4];  // m[src][dst]
        for (int s = 0; s < 4; ++s) {
            double mx = -1e300;
            for (int d = 0; d < 4; ++d) {
                const int idx = 8 + 4 * s + d;
                m[s][d]       = (double) mixes.p[idx + 24 * t] * scale.p[2] + base.p[idx];
                mx            = std::max(mx, m[s][d]);
            }
            double sum = 0.0;
            for (int d = 0; d < 4; ++d) {
                m[s][d] = std::exp(m[s][d] - mx);
                sum += m[s][d];
            }
            for (int d = 0; d < 4; ++d) {
                m[s][d] = m[s][d] / sum + eps;
            }
        }
        auto cols = [&] {
            for (int d = 0; d < 4; ++d) {
                double sum = eps;
                for (int s = 0; s < 4; ++s) {
                    sum += m[s][d];
                }
                for (int s = 0; s < 4; ++s) {
                    m[s][d] /= sum;
                }
            }
        };
        auto rows = [&] {
            for (int s = 0; s < 4; ++s) {
                double sum = eps;
                for (int d = 0; d < 4; ++d) {
                    sum += m[s][d];
                }
                for (int d = 0; d < 4; ++d) {
                    m[s][d] /= sum;
                }
            }
        };
        cols();
        for (int i = 1; i < n_iter; ++i) {
            rows();
            cols();
        }
        for (int s = 0; s < 4; ++s) {
            for (int d = 0; d < 4; ++d) {
                out[(size_t) (d + 4 * s + 16 * t)] = m[s][d];
            }
        }
    }
    return out;
}

void test_comb(sycl::queue & q, int64_t n_tokens, int n_iter, float eps) {
    buf mixes(q, 24 * (size_t) n_tokens);
    buf scale(q, 3);
    buf base(q, 24);
    buf d(q, 16 * (size_t) n_tokens);
    // logits spread over about +-8, so the softmax is peaky and Sinkhorn has real work to do; near-uniform
    // logits leave every normalisation an almost-no-op and no mutation of it would show
    mixes.fill(-2.0f, 2.0f);
    scale.fill(2.0f, 3.0f);
    base.fill(-1.0f, 1.0f);

    hc_comb_args a = {};
    a.mixes        = mixes.p;
    a.scale        = scale.p;
    a.base         = base.p;
    a.dst          = d.p;
    a.n_tokens     = n_tokens;
    a.sm0 = 1, a.sm1 = 24;
    a.ss0 = 1;
    a.sb0 = 1;
    a.sd0 = 1, a.sd1 = 4, a.sd2 = 16;
    a.eps    = eps;
    a.n_iter = n_iter;
    hc_comb_launch(q, a);
    q.wait_and_throw();

    const std::vector<double> got(d.p, d.p + d.n);
    const std::string         name = "comb tokens=" + std::to_string(n_tokens) + " n_iter=" + std::to_string(n_iter);
    expect_match(name, got, comb_oracle(mixes, n_tokens, scale, base, eps, n_iter));
    if (n_iter > 1) {
        // Sinkhorn converges within a few iterations, so "one iteration fewer" is not a different answer;
        // a kernel that never ran the row normalisation (n_iter == 1) is.
        expect_reject(name + " vs no row normalisation", got, comb_oracle(mixes, n_tokens, scale, base, eps, 1));
    }
}

// ---- DSV4_HC_POST ----------------------------------------------------------------------------------------------

std::vector<double> post_oracle(const buf &   x,
                                const view3 & vx,
                                const buf &   res,
                                const view3 & vr,
                                const buf &   post,
                                const view3 & vp,
                                const buf *   comb,
                                const view3 & vc,
                                bool          wrong_zero_comb = false) {
    const int64_t       n_embd = vx.n0, n_tokens = vx.n1, hc = vr.n1;
    std::vector<double> out;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t d = 0; d < hc; ++d) {
            for (int64_t i = 0; i < n_embd; ++i) {
                double sum = (double) x.p[vx.at(i, t, 0)] * post.p[vp.at(d, t, 0)];
                if (comb != nullptr) {
                    for (int64_t s = 0; s < hc; ++s) {
                        sum += (double) res.p[vr.at(i, s, t)] * comb->p[vc.at(d, s, t)];
                    }
                } else if (!wrong_zero_comb) {
                    sum += res.p[vr.at(i, d, t)];
                }
                out.push_back(sum);
            }
        }
    }
    return out;
}

void test_post(sycl::queue & q, int64_t n_embd, int64_t hc, int64_t n_tokens, bool identity, bool pad) {
    const view3 vx = pad ? view3{ n_embd, n_tokens, 1, 1, n_embd + 3, (n_embd + 3) * n_tokens } :
                           view3{ n_embd, n_tokens, 1, 1, n_embd, n_embd * n_tokens };
    const view3 vr = pad ? padded(n_embd, hc, n_tokens) : contiguous(n_embd, hc, n_tokens);
    const view3 vp =
        pad ? view3{ hc, n_tokens, 1, 1, hc + 1, (hc + 1) * n_tokens } : view3{ hc, n_tokens, 1, 1, hc, hc * n_tokens };
    const view3 vc = pad ? padded(hc, hc, n_tokens) : contiguous(hc, hc, n_tokens);
    const view3 vd = pad ? padded(n_embd, hc, n_tokens) : contiguous(n_embd, hc, n_tokens);

    buf x(q, vx.span());
    buf res(q, vr.span());
    buf post(q, vp.span());
    buf comb(q, vc.span());
    buf d(q, vd.span());
    x.fill(-1.0f, 1.0f);
    res.fill(-1.0f, 1.0f);
    post.fill(0.0f, 2.0f);
    comb.fill(0.0f, 1.0f);
    d.fill_value(12345.0f);

    hc_post_args a = {};
    a.x            = x.p;
    a.residual     = res.p;
    a.post         = post.p;
    a.comb         = identity ? nullptr : comb.p;
    a.dst          = d.p;
    a.n_embd       = n_embd;
    a.hc           = hc;
    a.n_tokens     = n_tokens;
    a.sx0 = vx.s0, a.sx1 = vx.s1;
    a.sr0 = vr.s0, a.sr1 = vr.s1, a.sr2 = vr.s2;
    a.sp0 = vp.s0, a.sp1 = vp.s1;
    a.sc0 = identity ? 0 : vc.s0, a.sc1 = identity ? 0 : vc.s1, a.sc2 = identity ? 0 : vc.s2;
    a.sd0 = vd.s0, a.sd1 = vd.s1, a.sd2 = vd.s2;
    hc_post_launch(q, a);
    q.wait_and_throw();

    std::vector<double> got;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t dd = 0; dd < hc; ++dd) {
            for (int64_t i = 0; i < n_embd; ++i) {
                got.push_back(d.p[vd.at(i, dd, t)]);
            }
        }
    }
    const std::string name = "post n_embd=" + std::to_string(n_embd) + " hc=" + std::to_string(hc) +
                             " tokens=" + std::to_string(n_tokens) + (identity ? " identity" : " comb") +
                             (pad ? " strided" : "");
    expect_match(name, got, post_oracle(x, vx, res, vr, post, vp, identity ? nullptr : &comb, vc));
    if (pad) {
        report(name + " leaves padding alone", padding_untouched(d, vd, 12345.0f), 0.0);
    }
    if (identity && n_embd * hc * n_tokens >= 64) {
        expect_reject("post identity vs zero comb", got, post_oracle(x, vx, res, vr, post, vp, nullptr, vc, true));
    }
}

// ---- LIGHTNING_INDEXER -----------------------------------------------------------------------------------------

struct lid_case {
    int64_t   n_embd, n_head, n_kv, n_batch, n_stream, nem3;
    ggml_type type;
    bool      pad          = false;  // strided q/k/w/m/dst with slack
    int64_t   max_groups_x = 0;      // launch grid width cap (0: the production default)
};

std::vector<double> lid_oracle(const std::vector<float> & q,
                               const std::vector<float> & kf,
                               const std::vector<float> & w,
                               const std::vector<float> & m,
                               const lid_case &           c,
                               bool                       wrong_no_relu = false,
                               bool                       wrong_no_mask = false) {
    std::vector<double> out;
    for (int64_t s = 0; s < c.n_stream; ++s) {
        for (int64_t t = 0; t < c.n_batch; ++t) {
            for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                double score = 0.0;
                for (int64_t h = 0; h < c.n_head; ++h) {
                    double dot = 0.0;
                    for (int64_t e = 0; e < c.n_embd; ++e) {
                        dot += (double) q[(size_t) (e + c.n_embd * (h + c.n_head * (t + c.n_batch * s)))] *
                               kf[(size_t) (e + c.n_embd * (ik + c.n_kv * s))];
                    }
                    score +=
                        (wrong_no_relu ? dot : std::max(dot, 0.0)) * w[(size_t) (h + c.n_head * (t + c.n_batch * s))];
                }
                const double mv = m[(size_t) (ik + c.n_kv * (t + c.n_batch * (s % c.nem3)))];
                out.push_back(wrong_no_mask ? score : score + mv);
            }
        }
    }
    return out;
}

template <int LANES> void test_lid(sycl::queue & q, const lid_case & c) {
    // K rows: random floats, stored in c.type with ggml-base's reference quantizer, decoded back by its to_float
    const ggml_type_traits * tr        = ggml_get_type_traits(c.type);
    const size_t             row_bytes = ggml_row_size(c.type, c.n_embd);
    std::vector<float>       kf((size_t) (c.n_embd * c.n_kv * c.n_stream));
    for (float & v : kf) {
        v = rnd(-1.0f, 1.0f);
    }
    std::vector<char> kq(row_bytes * (size_t) (c.n_kv * c.n_stream));
    for (int64_t r = 0; r < c.n_kv * c.n_stream; ++r) {
        char *  dstrow = kq.data() + row_bytes * (size_t) r;
        float * src    = kf.data() + (size_t) (c.n_embd * r);
        if (c.type == GGML_TYPE_F32) {
            std::memcpy(dstrow, src, row_bytes);
        } else {
            tr->from_float_ref(src, dstrow, c.n_embd);
            tr->to_float(dstrow, src, c.n_embd);  // the oracle reads what the kernel reads
        }
    }

    std::vector<float> qv((size_t) (c.n_embd * c.n_head * c.n_batch * c.n_stream));
    std::vector<float> wv((size_t) (c.n_head * c.n_batch * c.n_stream));
    std::vector<float> mv((size_t) (c.n_kv * c.n_batch * c.nem3));
    for (float & v : qv) {
        v = rnd(-1.0f, 1.0f);
    }
    for (float & v : wv) {
        v = rnd(0.0f, 1.0f);
    }
    for (float & v : mv) {
        // a causal-style mask: most keys visible, some -inf
        v = rnd(0.0f, 1.0f) < 0.2f ? -INFINITY : 0.0f;
    }

    // Strides in bytes. `pad` gives every row and plane the slack a view of a larger tensor has, so a kernel that
    // assumes packed strides, or reads or writes the slack, is caught.
    const size_t pq1 = c.pad ? 16 : 0, pq2 = c.pad ? 32 : 0, pq3 = c.pad ? 64 : 0;
    const size_t pk2 = c.pad ? 32 : 0, pk3 = c.pad ? 64 : 0;
    const size_t pw1 = c.pad ? 16 : 0, pw3 = c.pad ? 32 : 0;
    const size_t pm1 = c.pad ? 8 : 0, pm3 = c.pad ? 16 : 0;
    const size_t pd1 = c.pad ? 3 * sizeof(float) : 0, pd3 = c.pad ? 5 * sizeof(float) : 0;

    lightning_indexer_args a = {};
    a.k_type                 = c.type;
    a.n_embd                 = c.n_embd;
    a.n_head                 = c.n_head;
    a.n_batch                = c.n_batch;
    a.n_stream               = c.n_stream;
    a.n_kv                   = c.n_kv;
    a.nem3                   = c.nem3;
    a.max_groups_x           = c.max_groups_x;
    a.nbq1                   = sizeof(float) * c.n_embd + pq1;
    a.nbq2                   = a.nbq1 * c.n_head + pq2;
    a.nbq3                   = a.nbq2 * c.n_batch + pq3;
    a.nbk2                   = row_bytes + pk2;
    a.nbk3                   = a.nbk2 * c.n_kv + pk3;
    a.nbw1                   = sizeof(float) * c.n_head + pw1;
    a.nbw3                   = a.nbw1 * c.n_batch + pw3;
    a.nbm1                   = sizeof(sycl::half) * c.n_kv + pm1;
    a.nbm3                   = a.nbm1 * c.n_batch + pm3;
    a.nb1                    = sizeof(float) * c.n_kv + pd1;
    a.nb3                    = a.nb1 * c.n_batch + pd3;

    // whole buffers start as garbage (K as garbage blocks too), so only the strided view is meaningful
    auto alloc = [&](size_t bytes) {
        char * p = sycl::malloc_shared<char>(bytes ? bytes : 1, q);
        std::memset(p, 0x5a, bytes);
        return p;
    };
    char *       dq           = alloc(a.nbq3 * c.n_stream);
    char *       dk           = alloc(a.nbk3 * c.n_stream);
    char *       dw           = alloc(a.nbw3 * c.n_stream);
    char *       dm           = alloc(a.nbm3 * c.nem3);
    float *      dd           = (float *) alloc(a.nb3 * c.n_stream);
    const size_t n_dst_floats = a.nb3 * c.n_stream / sizeof(float);
    for (size_t i = 0; i < n_dst_floats; ++i) {
        dd[i] = 12345.0f;
    }

    for (int64_t s = 0; s < c.n_stream; ++s) {
        for (int64_t t = 0; t < c.n_batch; ++t) {
            for (int64_t h = 0; h < c.n_head; ++h) {
                std::memcpy(dq + s * a.nbq3 + t * a.nbq2 + h * a.nbq1,
                            qv.data() + (size_t) (c.n_embd * (h + c.n_head * (t + c.n_batch * s))),
                            sizeof(float) * c.n_embd);
            }
            std::memcpy(dw + s * a.nbw3 + t * a.nbw1, wv.data() + (size_t) (c.n_head * (t + c.n_batch * s)),
                        sizeof(float) * c.n_head);
        }
        for (int64_t ik = 0; ik < c.n_kv; ++ik) {
            std::memcpy(dk + s * a.nbk3 + ik * a.nbk2, kq.data() + row_bytes * (size_t) (ik + c.n_kv * s), row_bytes);
        }
    }
    for (int64_t sm = 0; sm < c.nem3; ++sm) {
        for (int64_t t = 0; t < c.n_batch; ++t) {
            for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                ((sycl::half *) (dm + sm * a.nbm3 + t * a.nbm1))[ik] =
                    (sycl::half) mv[(size_t) (ik + c.n_kv * (t + c.n_batch * sm))];
            }
        }
    }
    a.q = dq, a.k = dk, a.w = dw, a.m = dm, a.dst = dd;

    const bool launched = lightning_indexer_launch<LANES>(q, a);
    q.wait_and_throw();

    const std::string name = std::string("indexer lanes=") + std::to_string(LANES) +
                             " n_embd=" + std::to_string(c.n_embd) + " heads=" + std::to_string(c.n_head) +
                             " kv=" + std::to_string(c.n_kv) + " batch=" + std::to_string(c.n_batch) +
                             " streams=" + std::to_string(c.n_stream) + " mask_streams=" + std::to_string(c.nem3) +
                             " K=" + ggml_type_name(c.type) + (c.pad ? " strided" : "") +
                             (c.max_groups_x ? " groups_x<=" + std::to_string(c.max_groups_x) : "");
    if (!launched) {
        report(name + " launches", false, 0.0);
    } else {
        // the half mask rounds -inf/0 exactly, so the oracle may use the float mask
        std::vector<double> got;
        size_t              n_slack_touched = 0;
        std::vector<char>   seen(n_dst_floats, 0);
        for (int64_t s = 0; s < c.n_stream; ++s) {
            for (int64_t t = 0; t < c.n_batch; ++t) {
                for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                    const size_t at = (size_t) (ik + t * (a.nb1 / sizeof(float)) + s * (a.nb3 / sizeof(float)));
                    got.push_back(dd[at]);
                    seen[at] = 1;
                }
            }
        }
        for (size_t i = 0; i < n_dst_floats; ++i) {
            n_slack_touched += !seen[i] && dd[i] != 12345.0f;
        }
        expect_match(name, got, lid_oracle(qv, kf, wv, mv, c));
        if (c.pad) {
            report(name + " leaves padding alone", n_slack_touched == 0, (double) n_slack_touched);
        }
        if (c.n_kv * c.n_batch >= 16 && c.n_head >= 4) {
            expect_reject(name + " vs no ReLU", got, lid_oracle(qv, kf, wv, mv, c, true, false));
            expect_reject(name + " vs no mask", got, lid_oracle(qv, kf, wv, mv, c, false, true));
        }
    }

    sycl::free(dq, q);
    sycl::free(dk, q);
    sycl::free(dw, q);
    sycl::free(dm, q);
    sycl::free(dd, q);
}

// The grid arithmetic that keeps the launch inside 32-bit work-item ids. The shapes that overflowed a 1D range
// are too big to run here, so the arithmetic is checked on them directly, with the hardware-independent bound
// the compiler assumes (every global range dimension below 2^31, every group count below 2^31).
void test_lid_dims() {
    const int64_t INT_RANGE = int64_t(1) << 31;

    struct shape {
        const char * what;
        int64_t      n_rows;
    };

    const shape shapes[] = {
        { "empty",                                     0                      },
        { "one row",                                   1                      },
        { "one block",                                 4                      },
        { "one past a block",                          5                      },
        { "ub=512 kv=131072 (the no -c case)",         512LL * 131072         },
        { "ub=2048 kv=32768",                          2048LL * 32768         },
        { "2^26 - 1 rows (last 1D range that fit)",    (int64_t(1) << 26) - 1 },
        { "2^26 rows (first 1D overflow at 32 lanes)", int64_t(1) << 26       },
        { "ub=2048 kv=1048576",                        2048LL * 1048576       },
        { "4 streams of ub=512 kv=1048576",            4LL * 512 * 1048576    },
    };
    for (const shape & s : shapes) {
        const auto    d      = lightning_indexer_launch_dims(s.n_rows, 0);
        const int64_t groups = d.groups_x * d.groups_y;
        const bool    covers = s.n_rows == 0 ? d.groups_y == 0 : groups * LID_ROWS_PER_BLOCK >= s.n_rows;
        const bool    tight  = s.n_rows == 0 || (groups - d.groups_x) * LID_ROWS_PER_BLOCK < s.n_rows;
        for (int lanes : { 16, 32 }) {
            const bool fits = d.groups_x * LID_ROWS_PER_BLOCK * lanes < INT_RANGE && d.groups_y < INT_RANGE;
            report(std::string("indexer grid ") + s.what + " lanes=" + std::to_string(lanes) +
                       " covers every row, wastes under one grid row, every dimension fits 32-bit ids",
                   covers && tight && fits, (double) groups);
        }
    }
    // a 1D range for the overflow shapes WOULD have exceeded the id range: the control that makes the above bite
    report("control: a 1D range for 2^26 rows at 32 lanes exceeds the 32-bit id range",
           ((int64_t(1) << 26) + LID_ROWS_PER_BLOCK - 1) / LID_ROWS_PER_BLOCK * LID_ROWS_PER_BLOCK * 32 >= INT_RANGE,
           0.0);
}

bool sub_group_supported(const sycl::device & dev, size_t n) {
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        if (s == n) {
            return true;
        }
    }
    return false;
}

std::vector<sycl::device> cpu_devices() {
    std::vector<sycl::device> devs;
    try {
        devs = sycl::device::get_devices(sycl::info::device_type::cpu);
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL exception while enumerating CPU devices: %s\n", e.what());
    }
    return devs;
}

}  // namespace

int main() {
    const std::vector<sycl::device> devs = cpu_devices();
    if (devs.empty()) {
        fprintf(stderr,
                "SKIP: no SYCL CPU device visible; this run proves NOTHING about the DSv4 HC / indexer kernels.\n"
                "      Source oneAPI and run with ONEAPI_DEVICE_SELECTOR=opencl:cpu.\n");
        return SYCL_TEST_SKIP;
    }
    const sycl::device dev = devs[0];
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    sycl::queue q(dev, sycl::property::queue::in_order{});

    // --- PRE: Qwen3.8-Flash-Next emits gated hc=4 over n_embd=2560, one token (decode) and prefill batches
    test_pre(q, 2560, 4, 1, true, false);
    test_pre(q, 2560, 4, 37, true, false);
    test_pre(q, 2560, 4, 37, true, true);
    for (int64_t hc : { 1, 2, 3, 4, 5, 8, 65 }) {
        test_pre(q, 128, hc, 17, false, false);
        test_pre(q, 128, hc, 17, true, false);
    }
    test_pre(q, 1, 4, 1, false, false);
    test_pre(q, 31, 4, 17, false, false);
    test_pre(q, 31, 4, 17, false, true);
    test_pre(q, 4096, 4, 21, false, false);
    test_pre(q, 255, 4, 3, true, true);

    // --- COMB: hc is fixed at 4; production n_iter is 20; batches that straddle the 256-wide work-group
    for (int64_t n : { 1, 17, 255, 256, 257, 1024 }) {
        test_comb(q, n, 20, 1e-6f);
    }
    test_comb(q, 1, 1, 1e-6f);
    test_comb(q, 17, 4, 1e-6f);
    test_comb(q, 17, 2, 1e-6f);
    test_comb(q, 257, 8, 1e-3f);

    // --- POST: Qwen emits the identity (null comb) form; DeepSeek-V4 passes a real comb
    test_post(q, 2560, 4, 1, true, false);
    test_post(q, 2560, 4, 37, true, false);
    test_post(q, 2560, 4, 37, true, true);
    test_post(q, 2560, 4, 37, false, false);
    test_post(q, 31, 4, 17, false, true);
    test_post(q, 1, 4, 1, false, false);
    test_post(q, 1, 4, 1, true, false);
    test_post(q, 128, 3, 257, false, false);
    test_post(q, 128, 5, 19, true, true);
    test_post(q, 4096, 4, 21, false, false);

    // --- LIGHTNING_INDEXER: head size 128 (every model that carries an indexer) at the sub-group sizes the
    // device offers, K in every type the predicate admits, one and several streams, shared and per-stream masks
    const ggml_type k_types[] = { GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,  GGML_TYPE_Q5_1,
                                  GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL };
    int             lanes_run = 0;
    auto            run_lanes = [&](auto lanes_tag) {
        constexpr int LANES = decltype(lanes_tag)::value;
        if (!sub_group_supported(dev, LANES)) {
            printf("note: device has no sub-group size %d, indexer lanes=%d not run\n", LANES, LANES);
            return;
        }
        ++lanes_run;
        for (ggml_type t : k_types) {
            test_lid<LANES>(q, { 128, 4, 65, 32, 1, 1, t });
            test_lid<LANES>(q, { 128, 32, 7, 16, 4, 4, t });
            test_lid<LANES>(q, { 128, 4, 63, 9, 4, 1, t });
        }
        // strided operands, shared and per-stream masks, quantized and plain K
        test_lid<LANES>(q, { 128, 4, 33, 8, 1, 1, GGML_TYPE_F16, true });
        test_lid<LANES>(q, { 128, 4, 33, 8, 3, 3, GGML_TYPE_Q8_0, true });
        test_lid<LANES>(q, { 128, 4, 33, 8, 3, 1, GGML_TYPE_Q4_1, true });
        test_lid<LANES>(q, { 256, 2, 17, 5, 2, 2, GGML_TYPE_F32, true });
        // a tall grid (the width cap forces several rows of work-groups), packed and strided
        test_lid<LANES>(q, { 128, 4, 65, 32, 2, 1, GGML_TYPE_Q8_0, false, 3 });
        test_lid<LANES>(q, { 128, 4, 65, 32, 2, 2, GGML_TYPE_F16, true, 1 });
        test_lid<LANES>(q, { 128, 4, 1, 1, 1, 1, GGML_TYPE_F16, false, 1 });
        test_lid<LANES>(q, { 128, 4, 1, 1, 1, 1, GGML_TYPE_F16 });
        test_lid<LANES>(q, { 256, 4, 33, 8, 2, 2, GGML_TYPE_F16 });
        test_lid<LANES>(q, { 256, 4, 33, 8, 2, 1, GGML_TYPE_Q8_0 });
        test_lid<LANES>(q, { 64, 4, 33, 8, 2, 2, GGML_TYPE_F32 });
        if (LANES == 32) {  // 512 / 16 lanes = 32 elements per lane, which the kernel is not instantiated for
            test_lid<LANES>(q, { 512, 2, 17, 4, 1, 1, GGML_TYPE_F16 });
        }
    };
    test_lid_dims();
    run_lanes(std::integral_constant<int, 32>{});
    run_lanes(std::integral_constant<int, 16>{});
    ++g_cases;
    if (lanes_run == 0) {
        ++g_failed;
        printf("FAIL: the device offers neither sub-group size 16 nor 32, the indexer ran nowhere\n");
    } else {
        printf("PASS: the indexer ran at %d sub-group size(s)\n", lanes_run);
    }

    printf("\n%d checks, %d failed\n", g_cases, g_failed);
    if (g_cases < 100) {
        printf("FAIL: only %d checks ran; the matrix is not being exercised\n", g_cases);
        return 1;
    }
    return g_failed == 0 ? 0 : 1;
}
