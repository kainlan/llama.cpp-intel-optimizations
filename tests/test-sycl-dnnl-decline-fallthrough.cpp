// GPU test for llama.cpp-23mk S3-3 and S3-4 (design 4.8, G6): a declined oneDNN scratchpad is a decline, not an error.
//
// The three oneDNN wrappers outside ggml-sycl.cpp that ask for a scratchpad (softmax, eltwise, binary_broadcast_row)
// return false when the request is declined, before they write the op's output, and their callers fall through to the
// SYCL kernel the default path would have run. Every family measured so far asks for 0 bytes, so a real decline cannot
// be provoked from outside; the PRIVATE_TESTING seam ggml_sycl_test_inject_scratchpad_decline(site, after_n) forces one
// at the wrapper's decision and counts what each site did.
//
// Each arm but the last builds one small graph on one backend context and computes it twice, the seam off and then on
// (the last arm, below, computes once). The graph is computed directly on the SYCL backend with every tensor in a SYCL
// buffer. It must not go through ggml_backend_sched: the scheduler gives graph inputs to its last backend (the CPU),
// and the ops that consume them follow, so the first version of this test ran every op on the CPU and read
// calls == 0 at every site.
//
//   off  the wrapper must run: calls and engaged equal the arm's expected counts (1 and 1 for the single-call wrappers;
//        the KQ arms below count every launch of the graph), declined == 0, the output matches a host reference;
//   on   the wrapper must decline: calls == 1, declined == 1, engaged == 0 and the output still matches the host
//        reference (a fallback that ran over a half-written dst, or scaled a softmax twice, would not).
//
// Two further arms cover DnnlGemmWrapper::gemm (site "dnnl_gemm") through the batched f16 KQ mul_mat, which falls to a
// native GPU kernel on a decline: one with equal K and query head counts (one gemm per dim-3 slice) and one
// grouped-query arm (K has fewer heads) that takes the non-broadcast launch. That launch's hoisted pre-query is call 1
// of each slice's launch (the counters are cumulative across the graph, so slice s starts at call 5s + 1) and its
// per-pair gemm calls follow, so the counting statement "pre-query is call 1, batch b's own query is call b + 2" has a
// consumer.
//
// A decline after a write (a later call of the same launch, an inject with after_n > 1) throws
// dnnl_decline_after_write:dnnl_gemm, a ggml_sycl_fallback_error. The batched f16 caller in ggml_sycl_mul_mat_f16
// rethrows it (its `catch (const ggml_sycl_fallback_error &) { throw; }`), ggml_sycl_mul_mat rethrows it (its
// function-level catch of the same type) and ggml_backend_sycl_graph_compute turns it into GGML_STATUS_FAILED (its
// `catch (const ggml_sycl_fallback_error & error)`, which returns GGML_STATUS_FAILED). The last arm of main drives it
// on the grouped-query graph with after_n = 3 (call 1 the pre-query, call 2 the first gemm, which wrote dst, call 3
// the declined second gemm): graph_compute must return GGML_STATUS_FAILED, never SUCCESS and never a crash, with
// calls = 3, declined = 1, engaged = 1. Its output is undefined after the failed graph and is not compared; run_arm
// accepts a non-SUCCESS status only for an arm that sets expect_status. The other sites (MXFP4 PP, unified PP, MoE
// batched, the dense arms, out_prod) are pinned by scripts/check-sycl-dnnl-decline-consumers.py and have no device arm
// yet.
//
// The counters are the positive control. An arm whose off-run never reached its site (the env opt-in is missing, a
// shape fell under a threshold, the graph was recorded) has calls == 0 and FAILS as void; "identical" outputs from a
// run that never touched the wrapper would prove nothing. The oneDNN paths for SOFT_MAX and MUL are opt-in, so the
// registration sets GGML_SYCL_ONEDNN_SOFTMAX=1 and GGML_SYCL_ONEDNN_MUL=1, and GGML_SYCL_DISABLE_GRAPH=1 keeps the
// softmax and MUL arms out of graph recording (softmax.cpp and binbcast.cpp skip their oneDNN arm while recording).
//
// Outputs of the two runs are compared with each other and with a host reference. oneDNN and the SYCL kernels are
// different implementations, so only MUL (one IEEE multiply either way) is required to be bit-identical across the two
// runs; the others are held to the host reference within a tolerance and their bit equality is printed.
//
// Usage (the registration supplies the env; a bare run needs the three variables above):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ONEDNN_SOFTMAX=1 GGML_SYCL_ONEDNN_MUL=1 \
//     ./build/bin/test-sycl-dnnl-decline-fallthrough

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl-test.hpp"
#include "ggml-sycl.h"
#include "ggml.h"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#if !defined(GGML_SYCL_PRIVATE_TESTING)
#    error "this test requires GGML_SYCL_PRIVATE_TESTING (link against ggml-sycl-private-fixtures)"
#endif

namespace {

struct arm_graph {
    ggml_tensor * x   = nullptr;  // first input
    ggml_tensor * w   = nullptr;  // second input, MUL only
    ggml_tensor * out = nullptr;
};

struct arm {
    const char *                                            name;
    const char *                                            site;
    double                                                  tol;         // abs + rel tolerance against the host reference
    bool                                                    exact;       // off and on runs must be bit-identical
    ggml_type                                               w_type = GGML_TYPE_F32;  // the second input's storage type
    std::function<arm_graph(ggml_context *)>                build;
    std::function<void(std::vector<float> &, std::vector<float> &)> fill;  // x, w
    std::function<void(const std::vector<float> &, const std::vector<float> &, std::vector<float> &)> reference;

    // What the undeclined run must count at the site: calls the wrapper consulted it (a hoisted pre-query counts), engaged
    // the primitives it went on to submit. Derived from the arm's shape, not assumed to be 1.
    uint64_t expect_calls   = 1;
    uint64_t expect_engaged = 1;

    // The status graph_compute must return. Every arm but the post-write decline one expects SUCCESS.
    ggml_status expect_status = GGML_STATUS_SUCCESS;
};

constexpr int64_t SOFTMAX_COLS = 96;
constexpr int64_t SOFTMAX_ROWS = 160;  // oneDNN softmax needs >= 128 rows (softmax.cpp)
constexpr int64_t MUL_COLS     = 256;
constexpr int64_t MUL_ROWS     = 160;  // row-broadcast MUL needs a batch >= 128 (binbcast.cpp)
constexpr int64_t ELT_N        = 8192;  // the eltwise paths need >= 4096 elements (element_wise.cpp)
constexpr float   SOFTMAX_SCALE = 0.5f;
// The batched f16 mul_mat (a KQ-shaped graph: both operands permuted, one query column, more than one batch) reaches
// ggml_sycl_mul_mat_batched_sycl, whose oneDNN arm asks DnnlGemmWrapper::gemm (site "dnnl_gemm") once per launch. A
// decline falls to ggml_sycl_mul_mat_batched_f16_fallback, a native GPU kernel. K's batch dimension (dim 2) is strided, so
// the launches are made once per dim-3 slice, with the batch counts below:
//   KQ_HK == KQ_H  equal batch counts: one gemm per slice, so calls == engaged == KQ_B;
//   KQ_HK <  KQ_H  a grouped-query broadcast, the non-broadcast launch: one hoisted pre-query (call 1 of the slice,
//                  not engaged), then one gemm per (K head, query head per K head) pair, so per slice
//                  1 + KQ_H calls and KQ_H engaged.
constexpr int64_t KQ_D          = 64;
constexpr int64_t KQ_T          = 48;
constexpr int64_t KQ_H          = 4;  // query heads
constexpr int64_t KQ_HK         = 2;  // K heads of the grouped-query arm
constexpr int64_t KQ_B          = 2;

float input_value(int64_t i, int64_t n) {
    // a deterministic spread over [-6, 6) that is not a multiple pattern of any row width above
    return -6.0f + 12.0f * (float) ((i * 7919) % n) / (float) n;
}

std::vector<arm> make_arms() {
    std::vector<arm> arms;

    arms.push_back({ "SOFT_MAX scale 0.5, in place", "dnnl_softmax", 2e-5, false, GGML_TYPE_F32,
        [](ggml_context * ctx) {
            arm_graph g;
            g.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SOFTMAX_COLS, SOFTMAX_ROWS);
            ggml_set_input(g.x);
            // A scale by 1 in front, so the allocator may run the soft_max in place over its result: a declined
            // wrapper that had already pre-scaled that buffer would scale it twice.
            ggml_tensor * y = ggml_scale(ctx, g.x, 1.0f);
            g.out           = ggml_soft_max_ext_inplace(ctx, y, nullptr, SOFTMAX_SCALE, 0.0f);
            ggml_set_output(g.out);
            return g;
        },
        [](std::vector<float> & x, std::vector<float> &) {
            x.resize((size_t) (SOFTMAX_COLS * SOFTMAX_ROWS));
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = input_value((int64_t) i, (int64_t) x.size());
            }
        },
        [](const std::vector<float> & x, const std::vector<float> &, std::vector<float> & ref) {
            ref.resize(x.size());
            for (int64_t r = 0; r < SOFTMAX_ROWS; ++r) {
                double m = -1e30;
                for (int64_t c = 0; c < SOFTMAX_COLS; ++c) {
                    m = std::fmax(m, (double) x[(size_t) (r * SOFTMAX_COLS + c)] * SOFTMAX_SCALE);
                }
                double s = 0.0;
                for (int64_t c = 0; c < SOFTMAX_COLS; ++c) {
                    s += std::exp((double) x[(size_t) (r * SOFTMAX_COLS + c)] * SOFTMAX_SCALE - m);
                }
                for (int64_t c = 0; c < SOFTMAX_COLS; ++c) {
                    ref[(size_t) (r * SOFTMAX_COLS + c)] =
                        (float) (std::exp((double) x[(size_t) (r * SOFTMAX_COLS + c)] * SOFTMAX_SCALE - m) / s);
                }
            }
        } });

    arms.push_back({ "MUL row broadcast", "dnnl_binary_row", 0.0, true, GGML_TYPE_F32,
        [](ggml_context * ctx) {
            arm_graph g;
            g.x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MUL_COLS, MUL_ROWS);
            g.w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MUL_COLS, 1);
            ggml_set_input(g.x);
            ggml_set_input(g.w);
            g.out = ggml_mul(ctx, g.x, g.w);
            ggml_set_output(g.out);
            return g;
        },
        [](std::vector<float> & x, std::vector<float> & w) {
            x.resize((size_t) (MUL_COLS * MUL_ROWS));
            w.resize((size_t) MUL_COLS);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = input_value((int64_t) i, (int64_t) x.size());
            }
            for (size_t i = 0; i < w.size(); ++i) {
                w[i] = 0.25f + 0.0078125f * (float) i;
            }
        },
        [](const std::vector<float> & x, const std::vector<float> & w, std::vector<float> & ref) {
            ref.resize(x.size());
            for (int64_t r = 0; r < MUL_ROWS; ++r) {
                for (int64_t c = 0; c < MUL_COLS; ++c) {
                    ref[(size_t) (r * MUL_COLS + c)] = x[(size_t) (r * MUL_COLS + c)] * w[(size_t) c];
                }
            }
        } });

    const auto kq_arm = [&arms](const char * name, int64_t hk, uint64_t calls, uint64_t engaged) {
        arms.push_back(
            { name, "dnnl_gemm", 3e-2, false, GGML_TYPE_F16,
              [hk](ggml_context * ctx) {
                  arm_graph g;
                  // k and q are stored as [D, H, T|1, B] and permuted to [D, T|1, H, B], as llama's KQ is
                  g.x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KQ_D, KQ_H, 1, KQ_B);
                  g.w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KQ_D, hk, KQ_T, KQ_B);
                  ggml_set_input(g.x);
                  ggml_set_input(g.w);
                  g.out = ggml_mul_mat(ctx, ggml_permute(ctx, g.w, 0, 2, 1, 3), ggml_permute(ctx, g.x, 0, 2, 1, 3));
                  ggml_set_output(g.out);
                  return g;
              },
              [hk](std::vector<float> & q, std::vector<float> & k) {
                  q.resize((size_t) (KQ_D * KQ_H * KQ_B));
                  k.resize((size_t) (KQ_D * hk * KQ_T * KQ_B));
                  for (size_t i = 0; i < q.size(); ++i) {
                      q[i] = 0.2f * input_value((int64_t) i, (int64_t) q.size());
                  }
                  for (size_t i = 0; i < k.size(); ++i) {
                      k[i] = 0.2f * input_value((int64_t) i, (int64_t) k.size());
                  }
              },
              [hk](const std::vector<float> & q, const std::vector<float> & k, std::vector<float> & ref) {
                  ref.assign((size_t) (KQ_T * KQ_H * KQ_B), 0.0f);
                  for (int64_t b = 0; b < KQ_B; ++b) {
                      for (int64_t h = 0; h < KQ_H; ++h) {
                          const int64_t kh = h / (KQ_H / hk);  // query head h reads K head h / (heads per K head)
                          for (int64_t tt = 0; tt < KQ_T; ++tt) {
                              double s = 0.0;
                              for (int64_t d = 0; d < KQ_D; ++d) {
                                  const double kv = ggml_fp16_to_fp32(
                                      ggml_fp32_to_fp16(k[(size_t) (d + KQ_D * (kh + hk * (tt + KQ_T * b)))]));
                                  const double qv =
                                      ggml_fp16_to_fp32(ggml_fp32_to_fp16(q[(size_t) (d + KQ_D * (h + KQ_H * b))]));
                                  s += kv * qv;
                              }
                              ref[(size_t) (tt + KQ_T * (h + KQ_H * b))] = (float) s;
                          }
                      }
                  }
              },
              calls, engaged });
    };
    kq_arm("batched f16 KQ mul_mat", KQ_H, (uint64_t) KQ_B, (uint64_t) KQ_B);
    kq_arm("batched f16 KQ mul_mat GQA", KQ_HK, (uint64_t) KQ_B * (1 + KQ_H), (uint64_t) KQ_B * KQ_H);

    const auto eltwise = [&arms](const char * name, ggml_tensor * (*op)(ggml_context *, ggml_tensor *),
                                 double (*f)(double)) {
        arms.push_back({ name, "dnnl_eltwise", 5e-4, false, GGML_TYPE_F32,
            [op](ggml_context * ctx) {
                arm_graph g;
                g.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ELT_N);
                ggml_set_input(g.x);
                g.out = op(ctx, g.x);
                ggml_set_output(g.out);
                return g;
            },
            [](std::vector<float> & x, std::vector<float> &) {
                x.resize((size_t) ELT_N);
                for (size_t i = 0; i < x.size(); ++i) {
                    x[i] = input_value((int64_t) i, (int64_t) x.size());
                }
            },
            [f](const std::vector<float> & x, const std::vector<float> &, std::vector<float> & ref) {
                ref.resize(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    ref[i] = (float) f((double) x[i]);
                }
            } });
    };
    eltwise("SILU f32", ggml_silu, [](double x) { return x / (1.0 + std::exp(-x)); });
    eltwise("GELU f32", ggml_gelu,
            [](double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); });
    eltwise("GELU_ERF f32", ggml_gelu_erf, [](double x) { return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476)); });
    return arms;
}

struct run_result {
    bool               ok     = false;
    ggml_status        status = GGML_STATUS_FAILED;  // what graph_compute returned (set once it has returned)
    std::vector<float> out;
    uint64_t           calls = 0, declined = 0, engaged = 0;
};

// One compute of the arm's graph on the backend. inject != 0 forces a decline on that call of the site.
run_result run_arm(ggml_backend_t backend, const arm & a, int inject, bool keep_state = false, bool fresh = true) {
    run_result res;
    // fresh == false continues from the seam's current state: no reset and no new inject, so an armed after_n and the
    // counters carry over from the previous run_arm (which must have kept its state).
    if (fresh) {
        ggml_sycl_test_scratchpad_sites_reset();
        if (inject != 0 && !ggml_sycl_test_inject_scratchpad_decline(a.site, inject)) {
            fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", a.name, a.site);
            return res;
        }
    }

    const size_t         max_nodes = 16;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);

    ggml_init_params iparams = { /*.mem_size   =*/mem_size, /*.mem_buffer =*/mem_buffer.data(), /*.no_alloc   =*/true };
    ggml_context *   ctx     = ggml_init(iparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: %s: ggml_init failed\n", a.name);
        return res;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    arm_graph     g  = a.build(ctx);
    ggml_build_forward_expand(gf, g.out);

    // Every tensor, views included, lands in one SYCL buffer; an in-place op therefore really is in place.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_sycl_buffer_type(0));
    if (!buf) {
        fprintf(stderr, "FAIL: %s: could not allocate the graph's tensors on the SYCL device\n", a.name);
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    std::vector<float> x, w;
    a.fill(x, w);
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * sizeof(float));
    if (g.w && a.w_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> w16(w.size());
        ggml_fp32_to_fp16_row(w.data(), w16.data(), (int64_t) w.size());
        ggml_backend_tensor_set(g.w, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    } else if (g.w) {
        ggml_backend_tensor_set(g.w, w.data(), 0, w.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    res.status = status;
    if (status != a.expect_status) {
        fprintf(stderr, "FAIL: %s: graph compute returned %d, expected %d\n", a.name, (int) status,
                (int) a.expect_status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    // An expected failure (a decline after a write): the output is undefined after a failed graph, so it is not read,
    // and there is no recorded graph to check. Only the seam's counters are read.
    const bool expected_failure = a.expect_status != GGML_STATUS_SUCCESS;
    // Recording would route the softmax and MUL arms around their sites, so a recorded run is void.
    if (!expected_failure && ggml_sycl::test_backend_has_exec_graph(backend)) {
        fprintf(stderr, "FAIL: %s: the backend recorded an executable graph; the arm is void (run with GGML_SYCL_DISABLE_GRAPH=1)\n",
                a.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    if (!expected_failure) {
        res.out.resize((size_t) ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, res.out.data(), 0, res.out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    if (!ggml_sycl_test_scratchpad_site_counts(a.site, &res.calls, &res.declined, &res.engaged)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", a.name, a.site);
        return res;
    }
    if (!keep_state) {
        ggml_sycl_test_inject_scratchpad_decline(a.site, 0);
    }
    res.ok = true;
    return res;
}

bool within(const std::vector<float> & got, const std::vector<float> & ref, double tol, double * worst) {
    *worst = 0.0;
    if (got.size() != ref.size()) {
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        *worst         = std::fmax(*worst, d);
        if (!(d <= tol + tol * std::fabs((double) ref[i]))) {  // NaN fails too
            ok = false;
        }
    }
    return ok;
}

}  // namespace

int main(int, char ** argv) {
    // See tests/sycl-selector-fallback.hpp: a plain setenv() in main() is too late for libccl's static initializer.
    sycl_test_selector_fallback(argv, "level_zero:1");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr,
                "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    bool ok = true;
    // A negative after_n is not "never" and not "the first call": the seam refuses it, and an unknown site too.
    if (ggml_sycl_test_inject_scratchpad_decline("dnnl_softmax", -1) ||
        ggml_sycl_test_inject_scratchpad_decline("no_such_site", 1)) {
        fprintf(stderr, "FAIL: the seam accepted a negative after_n or an unknown site\n");
        ok = false;
    }
    ggml_sycl_test_scratchpad_sites_reset();
    {
        // A refused inject must change no state. Arm the first site, drive one call so the counters are nonzero, make a
        // refused inject (negative after_n), and expect the counters unchanged: a refusal that cleared them first would
        // read calls == 0 here.
        const std::vector<arm> arms0 = make_arms();
        const arm &            first = arms0.front();
        const run_result driven = run_arm(backend, first, 1, /*keep_state=*/true);
        uint64_t c = 0, d = 0, e = 0;
        const bool refused = !ggml_sycl_test_inject_scratchpad_decline(first.site, -1) &&
                             !ggml_sycl_test_inject_scratchpad_decline("no_such_site", 1);
        const bool read    = ggml_sycl_test_scratchpad_site_counts(first.site, &c, &d, &e);
        if (!driven.ok || driven.calls != 1 || driven.declined != 1 || !refused || !read || c != 1 || d != 1 || e != 0) {
            fprintf(stderr,
                    "FAIL: a refused inject changed the seam's state (driven calls=%llu declined=%llu; after the refusal "
                    "calls=%llu declined=%llu engaged=%llu, refused=%d)\n",
                    (unsigned long long) driven.calls, (unsigned long long) driven.declined, (unsigned long long) c,
                    (unsigned long long) d, (unsigned long long) e, refused ? 1 : 0);
            ok = false;
        }
        ggml_sycl_test_scratchpad_sites_reset();
    }
    {
        // A refusal must not disarm the site either. Arm the softmax site for its 2nd call, drive one call (no decline yet),
        // refuse two injects, drive a second call: it must be the declined one. A refusal that stored its after_n first
        // (or cleared the site) would let the second call through.
        const std::vector<arm> arms1 = make_arms();
        const arm &            first = arms1.front();
        ggml_sycl_test_scratchpad_sites_reset();
        const bool       armed   = ggml_sycl_test_inject_scratchpad_decline(first.site, 2);
        const run_result call1   = run_arm(backend, first, 0, /*keep_state=*/true, /*fresh=*/false);
        const bool       refused = !ggml_sycl_test_inject_scratchpad_decline(first.site, -1) &&
                                   !ggml_sycl_test_inject_scratchpad_decline("no_such_site", 1);
        const run_result call2   = run_arm(backend, first, 0, /*keep_state=*/true, /*fresh=*/false);
        if (!armed || !refused || !call1.ok || !call2.ok || call1.calls != 1 || call1.declined != 0 || call1.engaged != 1 ||
            call2.calls != 2 || call2.declined != 1 || call2.engaged != 1) {
            fprintf(stderr,
                    "FAIL: a refused inject disarmed the site or changed its counters (armed=%d refused=%d; call 1 calls=%llu "
                    "declined=%llu engaged=%llu; call 2 calls=%llu declined=%llu engaged=%llu)\n",
                    armed ? 1 : 0, refused ? 1 : 0, (unsigned long long) call1.calls, (unsigned long long) call1.declined,
                    (unsigned long long) call1.engaged, (unsigned long long) call2.calls, (unsigned long long) call2.declined,
                    (unsigned long long) call2.engaged);
            ok = false;
        }
        if (call2.ok) {
            // The declined second call fell through to the SYCL kernel: its output must match the host reference, as in the
            // per-arm loop below.
            std::vector<float> x, w, ref;
            first.fill(x, w);
            first.reference(x, w, ref);
            double worst = 0.0;
            if (!within(call2.out, ref, first.tol, &worst)) {
                fprintf(stderr, "FAIL: %s: the call declined after a refused inject differs from the host reference (max abs err %.3g)\n",
                        first.name, worst);
                ok = false;
            }
        }
        ggml_sycl_test_scratchpad_sites_reset();
    }
    for (const arm & a : make_arms()) {
        const run_result off = run_arm(backend, a, 0);
        const run_result on  = run_arm(backend, a, 1);
        if (!off.ok || !on.ok) {
            ok = false;
            continue;
        }
        std::vector<float> x, w, ref;
        a.fill(x, w);
        a.reference(x, w, ref);

        double worst_off = 0.0, worst_on = 0.0;
        const bool off_ok = within(off.out, ref, a.tol, &worst_off);
        const bool on_ok  = within(on.out, ref, a.tol, &worst_on);
        const bool same   = off.out.size() == on.out.size() &&
                            memcmp(off.out.data(), on.out.data(), off.out.size() * sizeof(float)) == 0;
        printf("%-30s site=%-16s off: calls=%llu declined=%llu engaged=%llu max_err=%.3g | on: calls=%llu declined=%llu "
               "engaged=%llu max_err=%.3g | bit_identical=%d\n",
               a.name, a.site, (unsigned long long) off.calls, (unsigned long long) off.declined,
               (unsigned long long) off.engaged, worst_off, (unsigned long long) on.calls,
               (unsigned long long) on.declined, (unsigned long long) on.engaged, worst_on, same ? 1 : 0);

        bool arm_ok = true;
        if (off.calls != a.expect_calls || off.engaged != a.expect_engaged || off.declined != 0) {
            fprintf(stderr,
                    "FAIL: %s: the undeclined run did not consult the site %llu time(s) and engage %llu (calls=%llu "
                    "engaged=%llu "
                    "declined=%llu); the arm is VOID or the call-counting statement is wrong -- check "
                    "GGML_SYCL_ONEDNN_SOFTMAX / "
                    "GGML_SYCL_ONEDNN_MUL and the shape thresholds\n",
                    a.name, (unsigned long long) a.expect_calls, (unsigned long long) a.expect_engaged,
                    (unsigned long long) off.calls, (unsigned long long) off.engaged,
                    (unsigned long long) off.declined);
            arm_ok = false;
        }
        if (on.calls != 1 || on.declined != 1 || on.engaged != 0) {
            fprintf(stderr,
                    "FAIL: %s: the forced decline did not fire exactly once and stop the wrapper (calls=%llu declined=%llu "
                    "engaged=%llu)\n",
                    a.name, (unsigned long long) on.calls, (unsigned long long) on.declined, (unsigned long long) on.engaged);
            arm_ok = false;
        }
        if (!off_ok) {
            fprintf(stderr, "FAIL: %s: the oneDNN run differs from the host reference (max abs err %.3g)\n", a.name, worst_off);
            arm_ok = false;
        }
        if (!on_ok) {
            fprintf(stderr, "FAIL: %s: the fallback after a decline differs from the host reference (max abs err %.3g)\n",
                    a.name, worst_on);
            arm_ok = false;
        }
        if (a.exact && !same) {
            fprintf(stderr, "FAIL: %s: the declined and undeclined runs are not bit-identical\n", a.name);
            arm_ok = false;
        }
        ok = ok && arm_ok;
    }

    {
        // A decline AFTER a write (llama.cpp-23mk S3-4, design 4.8): the GQA arm's first slice launches the hoisted
        // pre-query (call 1) and then one gemm per (K head, query head per K head) pair. Declining call 3 refuses the
        // second pair's gemm after the first pair wrote dst, which is not a next path: the decline throws
        // dnnl_decline_after_write:dnnl_gemm, a ggml_sycl_fallback_error that the caller, ggml_sycl_mul_mat and
        // compute_forward all pass through, and graph_compute returns GGML_STATUS_FAILED. It must not be SUCCESS (the
        // KQ product would be half-written and nobody told) and must not crash.
        //
        // The counters are the positive control: calls == 3 (pre-query, first gemm, declined gemm), declined == 1 and
        // engaged == 1 (the one gemm that wrote). The output buffer is NOT compared: after a failed graph it is
        // undefined. This arm runs last because a failed graph quarantines the backend's execution state.
        const char *           base_name = "batched f16 KQ mul_mat GQA";
        const std::vector<arm> base_arms = make_arms();
        const arm *            base      = nullptr;
        for (const arm & candidate : base_arms) {
            if (strcmp(candidate.name, base_name) == 0) {
                base = &candidate;
            }
        }
        if (!base) {
            // Never run a different arm in its place: its counters would not mean what this arm's expect.
            fprintf(stderr, "FAIL: arm '%s' not found: the post-write decline arm is built from it\n", base_name);
            ok = false;
        } else {
            arm post                 = *base;
            post.name                = "batched f16 KQ mul_mat GQA, post-write decline";
            post.expect_status       = GGML_STATUS_FAILED;
            const int        after_n = 3;
            const run_result r       = run_arm(backend, post, after_n);
            printf(
                "%-30s site=%-16s inject=%d: status=%d calls=%llu declined=%llu engaged=%llu (output not compared)\n",
                post.name, post.site, after_n, (int) r.status, (unsigned long long) r.calls,
                (unsigned long long) r.declined, (unsigned long long) r.engaged);
            if (!r.ok || r.status != GGML_STATUS_FAILED || r.calls != 3 || r.declined != 1 || r.engaged != 1) {
                fprintf(stderr,
                        "FAIL: %s: expected status FAILED with calls=3 declined=1 engaged=1 (got ok=%d status=%d "
                        "calls=%llu declined=%llu engaged=%llu)\n",
                        post.name, r.ok ? 1 : 0, (int) r.status, (unsigned long long) r.calls,
                        (unsigned long long) r.declined, (unsigned long long) r.engaged);
                ok = false;
            }
        }
    }

    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "test-sycl-dnnl-decline-fallthrough: FAIL\n");
        return 1;
    }
    printf("test-sycl-dnnl-decline-fallthrough: PASS\n");
    return 0;
}
