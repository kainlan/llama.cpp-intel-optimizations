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
#include "../dsv4-hc-predicates.hpp"
#include "../lightning-indexer-kernel.hpp"
#include "../lightning-indexer-predicate.hpp"
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
using namespace ggml_sycl_lightning_indexer;

namespace {

int g_cases  = 0;
int g_failed = 0;

constexpr double NMSE_TOL     = 1e-9;
constexpr double NMSE_REJECTS = 1e-4;  // a control oracle must be at least this far from the kernel

// The matrix below produces several hundred checks (about 480 with both sub-group sizes); a run
// that produces fewer than this had a loop silently skip its work, which a bare "no failure" would report as a pass.
constexpr int MIN_CHECKS = 100;

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

// The oracle's layout, from the op definition in ggml.h rather than from the kernel: per token, mixes[:, t] is
// [pre(4), post(4), comb(4 x 4)], so source stream s, destination d of the comb sits at HC_COMB_COMB_OFFSET + 4 s + d.
std::vector<double> comb_oracle(const buf &   mixes,
                                const view3 & vm,
                                const buf &   scale,
                                int64_t       ss0,
                                const buf &   base,
                                int64_t       sb0,
                                double        eps,
                                int           n_iter) {
    const int64_t       n_tokens = vm.n1;
    std::vector<double> out(16 * (size_t) n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        double m[4][4];  // m[src][dst]
        for (int s = 0; s < HC_COMB_STREAMS; ++s) {
            double mx = -1e300;
            for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                const int64_t idx = HC_COMB_COMB_OFFSET + HC_COMB_STREAMS * s + d;
                m[s][d] =
                    (double) mixes.p[vm.at(idx, t, 0)] * scale.p[HC_COMB_SCALE_COMB_IDX * ss0] + base.p[idx * sb0];
                mx = std::max(mx, m[s][d]);
            }
            double sum = 0.0;
            for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                m[s][d] = std::exp(m[s][d] - mx);
                sum += m[s][d];
            }
            for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                m[s][d] = m[s][d] / sum + eps;
            }
        }
        auto cols = [&] {
            for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                double sum = eps;
                for (int s = 0; s < HC_COMB_STREAMS; ++s) {
                    sum += m[s][d];
                }
                for (int s = 0; s < HC_COMB_STREAMS; ++s) {
                    m[s][d] /= sum;
                }
            }
        };
        auto rows = [&] {
            for (int s = 0; s < HC_COMB_STREAMS; ++s) {
                double sum = eps;
                for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                    sum += m[s][d];
                }
                for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                    m[s][d] /= sum;
                }
            }
        };
        cols();
        for (int i = 1; i < n_iter; ++i) {
            rows();
            cols();
        }
        for (int s = 0; s < HC_COMB_STREAMS; ++s) {
            for (int d = 0; d < HC_COMB_STREAMS; ++d) {
                out[(size_t) (d + HC_COMB_STREAMS * s + HC_COMB_STREAMS * HC_COMB_STREAMS * t)] = m[s][d];
            }
        }
    }
    return out;
}

// `pad` strides every operand the way a view of a larger tensor is strided: mixes, scale and base with an element
// stride of 2 plus slack between columns, dst with padded rows and planes. A kernel that assumes packed strides, or
// that touches the slack, fails the comparison or the padding check.
void test_comb(sycl::queue & q, int64_t n_tokens, int n_iter, float eps, bool pad = false) {
    constexpr int64_t SCALE_LEN = HC_COMB_SCALE_COMB_IDX + 1;
    const int64_t     elem      = pad ? 2 : 1;
    const view3       vm        = pad ? view3{ HC_COMB_MIX_DIM, n_tokens, 1, elem, HC_COMB_MIX_DIM * elem + 3, 0 } :
                                        view3{ HC_COMB_MIX_DIM, n_tokens, 1, 1, HC_COMB_MIX_DIM, 0 };
    const view3       vd        = pad ? padded(HC_COMB_STREAMS, HC_COMB_STREAMS, n_tokens) :
                                        contiguous(HC_COMB_STREAMS, HC_COMB_STREAMS, n_tokens);
    buf               mixes(q, vm.span());
    buf               scale(q, 1 + (SCALE_LEN - 1) * elem);
    buf               base(q, 1 + (HC_COMB_MIX_DIM - 1) * elem);
    buf               d(q, vd.span());
    // logits spread over about +-8, so the softmax is peaky and Sinkhorn has real work to do; near-uniform
    // logits leave every normalisation an almost-no-op and no mutation of it would show
    mixes.fill(-2.0f, 2.0f);
    scale.fill(2.0f, 3.0f);
    base.fill(-1.0f, 1.0f);
    d.fill_value(12345.0f);

    hc_comb_args a = {};
    a.mixes        = mixes.p;
    a.scale        = scale.p;
    a.base         = base.p;
    a.dst          = d.p;
    a.n_tokens     = n_tokens;
    a.sm0 = vm.s0, a.sm1 = vm.s1;
    a.ss0 = elem;
    a.sb0 = elem;
    a.sd0 = vd.s0, a.sd1 = vd.s1, a.sd2 = vd.s2;
    a.eps    = eps;
    a.n_iter = n_iter;
    hc_comb_launch(q, a);
    q.wait_and_throw();

    std::vector<double> got;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t s = 0; s < HC_COMB_STREAMS; ++s) {
            for (int64_t dd = 0; dd < HC_COMB_STREAMS; ++dd) {
                got.push_back(d.p[vd.at(dd, s, t)]);
            }
        }
    }
    const std::string name =
        "comb tokens=" + std::to_string(n_tokens) + " n_iter=" + std::to_string(n_iter) + (pad ? " strided" : "");
    expect_match(name, got, comb_oracle(mixes, vm, scale, elem, base, elem, eps, n_iter));
    if (pad) {
        report(name + " leaves padding alone", padding_untouched(d, vd, 12345.0f), 0.0);
    }
    if (n_iter > 1) {
        // Sinkhorn converges within a few iterations, so "one iteration fewer" is not a different answer;
        // a kernel that never ran the row normalisation (n_iter == 1) is.
        expect_reject(name + " vs no row normalisation", got, comb_oracle(mixes, vm, scale, elem, base, elem, eps, 1));
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

struct lightning_indexer_case {
    int64_t   n_embd, n_head, n_kv, n_batch, n_stream, nem3;
    ggml_type type;
    bool      pad          = false;  // strided q/k/w/m/dst with slack
    int64_t   max_groups_x = 0;      // launch grid width cap (0: the production default)
};

std::vector<double> lightning_indexer_oracle(const std::vector<float> &     q,
                                             const std::vector<float> &     kf,
                                             const std::vector<float> &     w,
                                             const std::vector<float> &     m,
                                             const lightning_indexer_case & c,
                                             bool                           wrong_no_relu = false,
                                             bool                           wrong_no_mask = false) {
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

template <int LANES> void test_lightning_indexer(sycl::queue & q, const lightning_indexer_case & c) {
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

    const bool launched = lightning_indexer_launch<LANES>(q, a, c.max_groups_x);
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
        expect_match(name, got, lightning_indexer_oracle(qv, kf, wv, mv, c));
        if (c.pad) {
            report(name + " leaves padding alone", n_slack_touched == 0, (double) n_slack_touched);
        }
        if (c.n_kv * c.n_batch >= 16 && c.n_head >= 4) {
            expect_reject(name + " vs no ReLU", got, lightning_indexer_oracle(qv, kf, wv, mv, c, true, false));
            expect_reject(name + " vs no mask", got, lightning_indexer_oracle(qv, kf, wv, mv, c, false, true));
        }
    }

    sycl::free(dq, q);
    sycl::free(dk, q);
    sycl::free(dw, q);
    sycl::free(dm, q);
    sycl::free(dd, q);
}

// launch returns true for exactly the (K type, n_embd) combinations the predicate admits at this lane count and
// false for every other, without touching the queue; nothing else asserts the `false` half, and a launcher that
// silently ran nothing for an unadmitted shape would leave its destination unwritten.
template <int LANES> void test_lightning_indexer_dispatch(sycl::queue & q) {
    int mismatches = 0;
    int admitted   = 0;
    int refused    = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        for (int64_t n_embd : { 16LL, 32LL, 48LL, 64LL, 128LL, 256LL, 512LL, 1024LL }) {
            lightning_indexer_args a = {};
            a.k_type                 = (ggml_type) t;
            a.n_embd                 = n_embd;  // every size/count is zero: an admitted launch is a no-op range
            const bool want = k_type_supported((ggml_type) t) && n_embd % LANES == 0 && epl_supported(n_embd / LANES);
            const bool got  = lightning_indexer_launch<LANES>(q, a);
            q.wait_and_throw();
            mismatches += got != want;
            admitted += got;
            refused += !got;
        }
    }
    report("indexer launch lanes=" + std::to_string(LANES) +
               " returns true for exactly the admitted (K type, n_embd) pairs",
           mismatches == 0, (double) mismatches);
    // non-vacuous: both halves of the assertion were exercised
    report("indexer launch lanes=" + std::to_string(LANES) + " dispatch matrix has admitted and refused pairs",
           admitted > 0 && refused > 0, (double) (admitted + refused));
}

// The grid arithmetic that keeps the launch inside 32-bit work-item ids. The shapes that overflowed a 1D range
// are too big to run here, so the arithmetic is checked on them directly, with the hardware-independent bound
// the compiler assumes (every global range dimension below 2^31, every group count below 2^31).
void test_lightning_indexer_dims() {
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
        const bool    covers = s.n_rows == 0 ? d.groups_y == 0 : groups * LIGHTNING_INDEXER_ROWS_PER_BLOCK >= s.n_rows;
        const bool    tight  = s.n_rows == 0 || (groups - d.groups_x) * LIGHTNING_INDEXER_ROWS_PER_BLOCK < s.n_rows;
        for (int lanes : { 16, 32 }) {
            const bool fits =
                d.groups_x * LIGHTNING_INDEXER_ROWS_PER_BLOCK * lanes < INT_RANGE && d.groups_y < INT_RANGE;
            report(std::string("indexer grid ") + s.what + " lanes=" + std::to_string(lanes) +
                       " covers every row, wastes under one grid row, every dimension fits 32-bit ids",
                   covers && tight && fits, (double) groups);
        }
    }
    // a 1D range for the overflow shapes WOULD have exceeded the id range: the control that makes the above bite
    report("control: a 1D range for 2^26 rows at 32 lanes exceeds the 32-bit id range",
           ((int64_t(1) << 26) + LIGHTNING_INDEXER_ROWS_PER_BLOCK - 1) / LIGHTNING_INDEXER_ROWS_PER_BLOCK *
                   LIGHTNING_INDEXER_ROWS_PER_BLOCK * 32 >=
               INT_RANGE,
           0.0);
}

// ---- supports_op predicates ------------------------------------------------------------------------------------
//
// The predicates ggml_backend_sycl_device_supports_op asks are pure ggml, so they run here over REAL op nodes built
// with the ggml API (shapes the API itself would assert on are reached by editing the built node, the way a
// view or a graph rewrite could present them). Each decline case changes exactly one property of an accepted node,
// so a predicate that dropped that one check turns the case red; each family also has accepting cases, so a
// predicate that declined everything is not a pass.

struct node_ctx {
    ggml_context * c;

    node_ctx() {
        ggml_init_params ip = {};
        ip.mem_size         = 4 * 1024 * 1024;
        ip.no_alloc         = true;
        c                   = ggml_init(ip);
    }

    ~node_ctx() { ggml_free(c); }

    ggml_tensor * t(ggml_type ty, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
        return ggml_new_tensor_4d(c, ty, n0, n1, n2, n3);
    }
};

void expect_pred(const std::string & name, bool got, bool want) {
    report(std::string("predicate: ") + name + (want ? " is admitted" : " is declined"), got == want, 0.0);
}

constexpr int64_t PRED_N_EMBD = 64;
constexpr int64_t PRED_N_TOK  = 3;
constexpr int     PRED_N_ITER = 20;
constexpr int     PRED_OP_PARAM_N_ITER =
    1;  // op_params slot ggml_dsv4_hc_comb stores n_iter in (and hc_pre its gated flag)

void test_predicates_pre() {
    auto build = [](node_ctx & n, bool gated) {
        ggml_tensor * x = n.t(GGML_TYPE_F32, PRED_N_EMBD, 4, PRED_N_TOK);
        return gated ? ggml_dsv4_hc_pre_gated(n.c, x, n.t(GGML_TYPE_F32, PRED_N_EMBD, 4, PRED_N_TOK), 1.0f) :
                       ggml_dsv4_hc_pre(n.c, x, n.t(GGML_TYPE_F32, 4, PRED_N_TOK));
    };

    struct mut {
        const char * what;
        bool         gated;
        void (*apply)(ggml_tensor *);
    };

    const mut muts[] = {
        { "x F16",                       false,
         [](ggml_tensor * op) {
              op->src[0]->type = GGML_TYPE_F16;
          } },
        { "w F16",                       false,
         [](ggml_tensor * op) {
              op->src[1]->type = GGML_TYPE_F16;
          } },
        { "dst F16",                     false,
         [](ggml_tensor * op) {
              op->type = GGML_TYPE_F16;
          } },
        { "x byte stride off a float",   false,
         [](ggml_tensor * op) {
              op->src[0]->nb[1] += 2;
          } },
        { "w byte stride off a float",   false,
         [](ggml_tensor * op) {
              op->src[1]->nb[1] += 2;
          } },
        { "dst byte stride off a float", false,
         [](ggml_tensor * op) {
              op->nb[1] += 2;
          } },
        { "x has a 4th dim",             false,
         [](ggml_tensor * op) {
              op->src[0]->ne[3] = 2;
          } },
        { "hc is zero",                  false,
         [](ggml_tensor * op) {
              op->src[0]->ne[1] = 0;
          } },
        { "dst n_embd wrong",            false,
         [](ggml_tensor * op) {
              op->ne[0] += 1;
          } },
        { "dst n_tokens wrong",          false,
         [](ggml_tensor * op) {
              op->ne[1] += 1;
          } },
        { "plain w hc wrong",            false,
         [](ggml_tensor * op) {
              op->src[1]->ne[0] += 1;
          } },
        { "plain w n_tokens wrong",      false,
         [](ggml_tensor * op) {
              op->src[1]->ne[1] += 1;
          } },
        { "gate n_embd wrong",           true,
         [](ggml_tensor * op) {
              op->src[1]->ne[0] += 1;
          } },
        { "gate hc wrong",               true,
         [](ggml_tensor * op) {
              op->src[1]->ne[1] += 1;
          } },
        { "gate n_tokens wrong",         true,
         [](ggml_tensor * op) {
              op->src[1]->ne[2] += 1;
          } },
    };
    for (bool gated : { false, true }) {
        node_ctx n;
        expect_pred(std::string("hc_pre ") + (gated ? "gated" : "plain"),
                    ggml_sycl_dsv4_hc_pre_supported(build(n, gated)), true);
    }
    for (const mut & m : muts) {
        node_ctx      n;
        ggml_tensor * op = build(n, m.gated);
        m.apply(op);
        expect_pred(std::string("hc_pre ") + (m.gated ? "gated " : "plain ") + m.what,
                    ggml_sycl_dsv4_hc_pre_supported(op), false);
    }
    expect_pred("hc_pre null op", ggml_sycl_dsv4_hc_pre_supported(nullptr), false);
    {
        node_ctx n;
        expect_pred("hc_pre fed a comb node", ggml_sycl_dsv4_hc_pre_supported([&] {
                        ggml_tensor * op = build(n, false);
                        op->op           = GGML_OP_DSV4_HC_COMB;
                        return op;
                    }()),
                    false);
    }
}

void test_predicates_comb() {
    auto build = [](node_ctx & n) {
        return ggml_dsv4_hc_comb(n.c, n.t(GGML_TYPE_F32, HC_COMB_MIX_DIM, PRED_N_TOK),
                                 n.t(GGML_TYPE_F32, HC_COMB_SCALE_COMB_IDX + 1), n.t(GGML_TYPE_F32, HC_COMB_MIX_DIM),
                                 1e-6f, PRED_N_ITER);
    };

    struct mut {
        const char * what;
        void (*apply)(ggml_tensor *);
    };

    const mut muts[] = {
        { "mixes F16",
         [](ggml_tensor * op) {
              op->src[0]->type = GGML_TYPE_F16;
          } },
        { "scale F16",
         [](ggml_tensor * op) {
              op->src[1]->type = GGML_TYPE_F16;
          } },
        { "base F16",
         [](ggml_tensor * op) {
              op->src[2]->type = GGML_TYPE_F16;
          } },
        { "dst F16",
         [](ggml_tensor * op) {
              op->type = GGML_TYPE_F16;
          } },
        { "mixes stride off a float",
         [](ggml_tensor * op) {
              op->src[0]->nb[1] += 2;
          } },
        { "hc != 4 (mixes dim 20)",
         [](ggml_tensor * op) {
              op->src[0]->ne[0] = 20;
          } },
        { "hc != 4 (base dim 20)",
         [](ggml_tensor * op) {
              op->src[2]->ne[0] = 20;
          } },
        { "hc != 4 (dst 3x3)",
         [](ggml_tensor * op) {
              op->ne[0] = 3, op->ne[1] = 3;
          } },
        { "scale too short for the comb scale",
         [](ggml_tensor * op) {
              op->src[1]->ne[0] = HC_COMB_SCALE_COMB_IDX;
          } },
        { "mixes has a plane",
         [](ggml_tensor * op) {
              op->src[0]->ne[2] = 2;
          } },
        { "dst n_tokens wrong",
         [](ggml_tensor * op) {
              op->ne[2] += 1;
          } },
        { "n_iter zero",
         [](ggml_tensor * op) {
              const int32_t zero = 0;
              std::memcpy((int32_t *) op->op_params + PRED_OP_PARAM_N_ITER, &zero, sizeof(zero));
          } },
    };
    {
        node_ctx n;
        expect_pred("hc_comb", ggml_sycl_dsv4_hc_comb_supported(build(n)), true);
    }
    {
        node_ctx      n;
        ggml_tensor * op = build(n);
        op->src[1]->ne[0] += 5;  // a longer scale vector is fine: only element HC_COMB_SCALE_COMB_IDX is read
        expect_pred("hc_comb with a longer scale vector", ggml_sycl_dsv4_hc_comb_supported(op), true);
    }
    for (const mut & m : muts) {
        node_ctx      n;
        ggml_tensor * op = build(n);
        m.apply(op);
        expect_pred(std::string("hc_comb ") + m.what, ggml_sycl_dsv4_hc_comb_supported(op), false);
    }
    expect_pred("hc_comb null op", ggml_sycl_dsv4_hc_comb_supported(nullptr), false);
}

void test_predicates_post() {
    auto build = [](node_ctx & n, bool with_comb) {
        return ggml_dsv4_hc_post(n.c, n.t(GGML_TYPE_F32, PRED_N_EMBD, PRED_N_TOK),
                                 n.t(GGML_TYPE_F32, PRED_N_EMBD, 4, PRED_N_TOK), n.t(GGML_TYPE_F32, 4, PRED_N_TOK),
                                 with_comb ? n.t(GGML_TYPE_F32, 4, 4, PRED_N_TOK) : nullptr);
    };

    struct mut {
        const char * what;
        bool         with_comb;
        void (*apply)(ggml_tensor *);
    };

    const mut muts[] = {
        { "x F16",                   false,
         [](ggml_tensor * op) {
              op->src[0]->type = GGML_TYPE_F16;
          } },
        { "residual F16",            false,
         [](ggml_tensor * op) {
              op->src[1]->type = GGML_TYPE_F16;
          } },
        { "post F16",                false,
         [](ggml_tensor * op) {
              op->src[2]->type = GGML_TYPE_F16;
          } },
        { "dst F16",                 false,
         [](ggml_tensor * op) {
              op->type = GGML_TYPE_F16;
          } },
        { "x stride off a float",    false,
         [](ggml_tensor * op) {
              op->src[0]->nb[1] += 2;
          } },
        { "x has a plane",           false,
         [](ggml_tensor * op) {
              op->src[0]->ne[2] = 2;
          } },
        { "hc is zero",              false,
         [](ggml_tensor * op) {
              op->src[1]->ne[1] = 0;
          } },
        { "residual n_embd wrong",   false,
         [](ggml_tensor * op) {
              op->src[1]->ne[0] += 1;
          } },
        { "post hc wrong",           false,
         [](ggml_tensor * op) {
              op->src[2]->ne[0] += 1;
          } },
        { "dst hc wrong",            false,
         [](ggml_tensor * op) {
              op->ne[1] += 1;
          } },
        { "comb F16",                true,
         [](ggml_tensor * op) {
              op->src[3]->type = GGML_TYPE_F16;
          } },
        { "comb stride off a float", true,
         [](ggml_tensor * op) {
              op->src[3]->nb[1] += 2;
          } },
        { "comb dim 0 wrong",        true,
         [](ggml_tensor * op) {
              op->src[3]->ne[0] = 3;
          } },
        { "comb dim 1 wrong",        true,
         [](ggml_tensor * op) {
              op->src[3]->ne[1] = 3;
          } },
        { "comb n_tokens wrong",     true,
         [](ggml_tensor * op) {
              op->src[3]->ne[2] += 1;
          } },
        { "comb has a 4th dim",      true,
         [](ggml_tensor * op) {
              op->src[3]->ne[3] = 2;
          } },
    };
    for (bool with_comb : { false, true }) {
        node_ctx n;
        expect_pred(std::string("hc_post ") + (with_comb ? "with comb" : "with a null comb (identity)"),
                    ggml_sycl_dsv4_hc_post_supported(build(n, with_comb)), true);
    }
    for (const mut & m : muts) {
        node_ctx      n;
        ggml_tensor * op = build(n, m.with_comb);
        m.apply(op);
        expect_pred(std::string("hc_post ") + (m.with_comb ? "with comb: " : "null comb: ") + m.what,
                    ggml_sycl_dsv4_hc_post_supported(op), false);
    }
    expect_pred("hc_post null op", ggml_sycl_dsv4_hc_post_supported(nullptr), false);
}

constexpr int64_t PRED_LANES = 32;  // the sub-group size the backend launches the indexer with

void test_predicates_indexer() {
    auto build = [](node_ctx & n, ggml_type kt, int64_t n_embd) {
        return ggml_lightning_indexer(n.c, n.t(GGML_TYPE_F32, n_embd, 4, 8, 2), n.t(kt, n_embd, 1, 16, 2),
                                      n.t(GGML_TYPE_F32, 4, 8, 1, 2), n.t(GGML_TYPE_F16, 16, 8, 1, 1));
    };

    struct mut {
        const char * what;
        void (*apply)(ggml_tensor *);
    };

    // the K operand is F16 in every mutation case, so mutations on k are about k alone
    const mut muts[] = {
        { "q F16",
         [](ggml_tensor * op) {
              op->src[0]->type = GGML_TYPE_F16;
          } },
        { "w F16",
         [](ggml_tensor * op) {
              op->src[2]->type = GGML_TYPE_F16;
          } },
        { "mask F32",
         [](ggml_tensor * op) {
              op->src[3]->type = GGML_TYPE_F32;
          } },
        { "mask BF16",
         [](ggml_tensor * op) {
              op->src[3]->type = GGML_TYPE_BF16;
          } },
        { "dst F16",
         [](ggml_tensor * op) {
              op->type = GGML_TYPE_F16;
          } },
        { "q nb[0] not one element",
         [](ggml_tensor * op) {
              op->src[0]->nb[0] = 8;
          } },
        { "k nb[0] not one element",
         [](ggml_tensor * op) {
              op->src[1]->nb[0] = 4;
          } },
        { "w nb[0] not one element",
         [](ggml_tensor * op) {
              op->src[2]->nb[0] = 8;
          } },
        { "mask nb[0] not one element",
         [](ggml_tensor * op) {
              op->src[3]->nb[0] = 4;
          } },
        { "dst nb[0] not one element",
         [](ggml_tensor * op) {
              op->nb[0] = 8;
          } },
        { "q n_embd != k n_embd",
         [](ggml_tensor * op) {
              op->src[1]->ne[0] += 2;
          } },
        { "k has a dim-1 extent",
         [](ggml_tensor * op) {
              op->src[1]->ne[1] = 2;
          } },
        { "k streams != q streams",
         [](ggml_tensor * op) {
              op->src[1]->ne[3] = 1;
          } },
        { "w n_head wrong",
         [](ggml_tensor * op) {
              op->src[2]->ne[0] += 1;
          } },
        { "w n_batch wrong",
         [](ggml_tensor * op) {
              op->src[2]->ne[1] += 1;
          } },
        { "w has a dim-2 extent",
         [](ggml_tensor * op) {
              op->src[2]->ne[2] = 2;
          } },
        { "mask n_kv wrong",
         [](ggml_tensor * op) {
              op->src[3]->ne[0] += 1;
          } },
        { "mask n_batch wrong",
         [](ggml_tensor * op) {
              op->src[3]->ne[1] += 1;
          } },
        { "mask streams do not divide",
         [](ggml_tensor * op) {
              op->src[3]->ne[3] = 3;
          } },
        { "mask streams zero",
         [](ggml_tensor * op) {
              op->src[3]->ne[3] = 0;
          } },
        { "dst n_kv wrong",
         [](ggml_tensor * op) {
              op->ne[0] += 1;
          } },
        { "dst n_batch wrong",
         [](ggml_tensor * op) {
              op->ne[1] += 1;
          } },
        { "dst stream count wrong",
         [](ggml_tensor * op) {
              op->ne[3] += 1;
          } },
        { "q nb[1] off a float",
         [](ggml_tensor * op) {
              op->src[0]->nb[1] += 2;
          } },
        { "q nb[2] off a float",
         [](ggml_tensor * op) {
              op->src[0]->nb[2] += 2;
          } },
        { "q nb[3] off a float",
         [](ggml_tensor * op) {
              op->src[0]->nb[3] += 2;
          } },
        { "w nb[1] off a float",
         [](ggml_tensor * op) {
              op->src[2]->nb[1] += 2;
          } },
        { "w nb[3] off a float",
         [](ggml_tensor * op) {
              op->src[2]->nb[3] += 2;
          } },
        { "mask nb[1] odd",
         [](ggml_tensor * op) {
              op->src[3]->nb[1] += 1;
          } },
        { "mask nb[3] odd",
         [](ggml_tensor * op) {
              op->src[3]->nb[3] += 1;
          } },
        { "k nb[2] odd",
         [](ggml_tensor * op) {
              op->src[1]->nb[2] += 1;
          } },
        { "k nb[3] odd",
         [](ggml_tensor * op) {
              op->src[1]->nb[3] += 1;
          } },
        { "dst nb[1] off a float",
         [](ggml_tensor * op) {
              op->nb[1] += 2;
          } },
        { "dst nb[3] off a float",
         [](ggml_tensor * op) {
              op->nb[3] += 2;
          } },
    };
    // slack that the kernel's casts tolerate: a half-aligned mask stride and a half-aligned F16 K stride
    const mut tolerated[] = {
        { "mask nb[1] 2-byte slack",
         [](ggml_tensor * op) {
              op->src[3]->nb[1] += 2;
          } },
        { "mask nb[3] 2-byte slack",
         [](ggml_tensor * op) {
              op->src[3]->nb[3] += 2;
          } },
        { "F16 k nb[2] 2-byte slack",
         [](ggml_tensor * op) {
              op->src[1]->nb[2] += 2;
          } },
        { "F16 k nb[3] 2-byte slack",
         [](ggml_tensor * op) {
              op->src[1]->nb[3] += 2;
          } },
        { "q nb[1] 4-byte slack",
         [](ggml_tensor * op) {
              op->src[0]->nb[1] += 4;
          } },
    };

    // every K type in the table is admitted at the production head size, in every elements-per-lane the kernel has
    for (ggml_type kt :
#define X(T) T,
         { GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X) }
#undef X
    ) {
        node_ctx n;
        expect_pred(std::string("indexer K=") + ggml_type_name(kt) + " n_embd=128",
                    op_supported(build(n, kt, 128), PRED_LANES), true);
    }
    for (int64_t n_embd : { 64, 128, 256, 512 }) {  // 2, 4, 8, 16 elements per lane at 32 lanes
        node_ctx n;
        expect_pred("indexer n_embd=" + std::to_string(n_embd) + " at 32 lanes",
                    op_supported(build(n, GGML_TYPE_F16, n_embd), 32), true);
    }
    {
        node_ctx n;
        expect_pred("indexer n_embd=128 at 16 lanes", op_supported(build(n, GGML_TYPE_F16, 128), 16), true);
    }
    for (const mut & m : tolerated) {
        node_ctx      n;
        ggml_tensor * op = build(n, GGML_TYPE_F16, 128);
        m.apply(op);
        expect_pred(std::string("indexer ") + m.what, op_supported(op, PRED_LANES), true);
    }

    // every K type outside the table is declined, enumerated from ggml's own type list rather than a list here
    int declined_types = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const int64_t blck = ggml_blck_size((ggml_type) t);
        if (k_type_supported((ggml_type) t) || blck <= 0 || ggml_type_size((ggml_type) t) == 0 || 256 % blck != 0) {
            continue;
        }
        node_ctx n;
        expect_pred(std::string("indexer K=") + ggml_type_name((ggml_type) t),
                    op_supported(build(n, (ggml_type) t, 256), PRED_LANES), false);
        ++declined_types;
    }
    report("predicate: the enumeration of unsupported K types is not empty", declined_types > 0,
           (double) declined_types);

    // K types whose block alignment differs from the half: 4-byte-aligned (Q4_1, Q5_1, F32) refuse the 2-byte slack
    for (ggml_type kt : { GGML_TYPE_F32, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1 }) {
        node_ctx      n;
        ggml_tensor * op = build(n, kt, 128);
        op->src[1]->nb[2] += 2;
        expect_pred(std::string("indexer K=") + ggml_type_name(kt) + " k nb[2] 2-byte slack",
                    op_supported(op, PRED_LANES), false);
    }
    {
        node_ctx      n;
        ggml_tensor * op = build(n, GGML_TYPE_Q8_0, 128);
        op->src[1]->nb[2] += 2;
        expect_pred("indexer K=q8_0 k nb[2] 2-byte slack", op_supported(op, PRED_LANES), true);
    }

    for (const mut & m : muts) {
        node_ctx      n;
        ggml_tensor * op = build(n, GGML_TYPE_F16, 128);
        m.apply(op);
        expect_pred(std::string("indexer ") + m.what, op_supported(op, PRED_LANES), false);
    }

    // head-size / lane arithmetic: n_embd % lanes, and an elements-per-lane count the kernel is not instantiated for
    for (int64_t n_embd : { 48, 96 }) {  // F32 K so the head size is not a block-size question
        node_ctx n;
        expect_pred("indexer n_embd=" + std::to_string(n_embd) + " (not a multiple of 32 lanes)",
                    op_supported(build(n, GGML_TYPE_F32, n_embd), 32), false);
    }
    {
        node_ctx n;
        expect_pred("indexer n_embd=1024 at 32 lanes (32 elements per lane)",
                    op_supported(build(n, GGML_TYPE_F16, 1024), 32), false);
    }
    {
        node_ctx n;
        expect_pred("indexer n_embd=512 at 16 lanes (32 elements per lane)",
                    op_supported(build(n, GGML_TYPE_F16, 512), 16), false);
    }
    {
        node_ctx n;
        expect_pred("indexer n_embd=32 at 32 lanes (1 element per lane)", op_supported(build(n, GGML_TYPE_F16, 32), 32),
                    false);
    }
    {
        node_ctx n;
        expect_pred("indexer lanes=0", op_supported(build(n, GGML_TYPE_F16, 128), 0), false);
    }
    expect_pred("indexer null op", op_supported(nullptr, PRED_LANES), false);
    {
        node_ctx      n;
        ggml_tensor * op = build(n, GGML_TYPE_F16, 128);
        op->src[3]       = nullptr;
        expect_pred("indexer null mask", op_supported(op, PRED_LANES), false);
    }
    {
        node_ctx      n;
        ggml_tensor * op = build(n, GGML_TYPE_F16, 128);
        op->op           = GGML_OP_DSV4_HC_PRE;
        expect_pred("indexer fed another op", op_supported(op, PRED_LANES), false);
    }
}

void test_predicates() {
    test_predicates_pre();
    test_predicates_comb();
    test_predicates_post();
    test_predicates_indexer();
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
    // pure ggml, no device: these run (and can fail the run) even where the kernels below are skipped
    test_predicates();

    const std::vector<sycl::device> devs = cpu_devices();
    if (devs.empty()) {
        fprintf(stderr,
                "SKIP: no SYCL CPU device visible; this run proves NOTHING about the DSv4 HC / indexer kernels.\n"
                "      Source oneAPI and run with ONEAPI_DEVICE_SELECTOR=opencl:cpu.\n");
        return g_failed == 0 ? SYCL_TEST_SKIP : 1;
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
    test_comb(q, 17, 20, 1e-6f, true);
    test_comb(q, 257, 4, 1e-6f, true);

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
    // generated from the predicate's own table, so a type added there is tested here without an edit
    const ggml_type k_types[] = {
#define X(T) T,
        GGML_SYCL_LIGHTNING_INDEXER_K_TYPES(X)
#undef X
    };
    int  lanes_run = 0;
    auto run_lanes = [&](auto lanes_tag) {
        constexpr int LANES = decltype(lanes_tag)::value;
        if (!sub_group_supported(dev, LANES)) {
            printf("note: device has no sub-group size %d, indexer lanes=%d not run\n", LANES, LANES);
            return;
        }
        ++lanes_run;
        for (ggml_type t : k_types) {
            test_lightning_indexer<LANES>(q, { 128, 4, 65, 32, 1, 1, t });
            test_lightning_indexer<LANES>(q, { 128, 32, 7, 16, 4, 4, t });
            test_lightning_indexer<LANES>(q, { 128, 4, 63, 9, 4, 1, t });
        }
        // strided operands, shared and per-stream masks, quantized and plain K
        test_lightning_indexer<LANES>(q, { 128, 4, 33, 8, 1, 1, GGML_TYPE_F16, true });
        test_lightning_indexer<LANES>(q, { 128, 4, 33, 8, 3, 3, GGML_TYPE_Q8_0, true });
        test_lightning_indexer<LANES>(q, { 128, 4, 33, 8, 3, 1, GGML_TYPE_Q4_1, true });
        test_lightning_indexer<LANES>(q, { 256, 2, 17, 5, 2, 2, GGML_TYPE_F32, true });
        // a tall grid (the width cap forces several rows of work-groups), packed and strided
        test_lightning_indexer<LANES>(q, { 128, 4, 65, 32, 2, 1, GGML_TYPE_Q8_0, false, 3 });
        test_lightning_indexer<LANES>(q, { 128, 4, 65, 32, 2, 2, GGML_TYPE_F16, true, 1 });
        test_lightning_indexer<LANES>(q, { 128, 4, 1, 1, 1, 1, GGML_TYPE_F16, false, 1 });
        test_lightning_indexer<LANES>(q, { 128, 4, 1, 1, 1, 1, GGML_TYPE_F16 });
        test_lightning_indexer<LANES>(q, { 256, 4, 33, 8, 2, 2, GGML_TYPE_F16 });
        test_lightning_indexer<LANES>(q, { 256, 4, 33, 8, 2, 1, GGML_TYPE_Q8_0 });
        test_lightning_indexer<LANES>(q, { 64, 4, 33, 8, 2, 2, GGML_TYPE_F32 });
        // n_embd 512 is 16 elements per lane at 32 lanes, an instantiated count; at 16 lanes it would be 32,
        // which is not (the launcher refuses it, checked in test_lightning_indexer_dispatch), so only 32 lanes run it
        if (LANES == 32) {
            test_lightning_indexer<LANES>(q, { 512, 2, 17, 4, 1, 1, GGML_TYPE_F16 });
        }
        test_lightning_indexer_dispatch<LANES>(q);
    };
    test_lightning_indexer_dims();
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
    if (g_cases < MIN_CHECKS) {
        printf("FAIL: only %d checks ran, fewer than the %d the matrix produces; it is not being exercised\n", g_cases,
               MIN_CHECKS);
        return 1;
    }
    return g_failed == 0 ? 0 : 1;
}
