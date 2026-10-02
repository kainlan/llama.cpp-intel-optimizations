// GPU test for llama.cpp-23mk S3-3 (design 4.8, G6): a declined oneDNN scratchpad is a decline, not an error.
//
// The three oneDNN wrappers outside ggml-sycl.cpp that ask for a scratchpad (softmax, eltwise, binary_broadcast_row)
// return false when the request is declined, before they write the op's output, and their callers fall through to the
// SYCL kernel the default path would have run. Every family measured so far asks for 0 bytes, so a real decline cannot
// be provoked from outside; the PRIVATE_TESTING seam ggml_sycl_test_inject_scratchpad_decline(site, after_n) forces one
// at the wrapper's decision and counts what each site did.
//
// Each arm builds one small graph on one backend context and computes it twice, the seam off and then on. The graph
// is computed directly on the SYCL backend with every tensor in a SYCL buffer. It must not go through
// ggml_backend_sched: the scheduler gives graph inputs to its last backend (the CPU), and the ops that consume them
// follow, so the first version of this test ran every op on the CPU and read calls == 0 at every site.
//
//   off  the wrapper must run: calls == 1, declined == 0, engaged == 1, the output matches a host reference;
//   on   the wrapper must decline: calls == 1, declined == 1, engaged == 0 and the output still matches the host
//        reference (a fallback that ran over a half-written dst, or scaled a softmax twice, would not).
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
    std::function<arm_graph(ggml_context *)>                build;
    std::function<void(std::vector<float> &, std::vector<float> &)> fill;  // x, w
    std::function<void(const std::vector<float> &, const std::vector<float> &, std::vector<float> &)> reference;
};

constexpr int64_t SOFTMAX_COLS = 96;
constexpr int64_t SOFTMAX_ROWS = 160;  // oneDNN softmax needs >= 128 rows (softmax.cpp)
constexpr int64_t MUL_COLS     = 256;
constexpr int64_t MUL_ROWS     = 160;  // row-broadcast MUL needs a batch >= 128 (binbcast.cpp)
constexpr int64_t ELT_N        = 8192;  // the eltwise paths need >= 4096 elements (element_wise.cpp)
constexpr float   SOFTMAX_SCALE = 0.5f;

float input_value(int64_t i, int64_t n) {
    // a deterministic spread over [-6, 6) that is not a multiple pattern of any row width above
    return -6.0f + 12.0f * (float) ((i * 7919) % n) / (float) n;
}

std::vector<arm> make_arms() {
    std::vector<arm> arms;

    arms.push_back({ "SOFT_MAX scale 0.5, in place", "dnnl_softmax", 2e-5, false,
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

    arms.push_back({ "MUL row broadcast", "dnnl_binary_row", 0.0, true,
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

    const auto eltwise = [&arms](const char * name, ggml_tensor * (*op)(ggml_context *, ggml_tensor *),
                                 double (*f)(double)) {
        arms.push_back({ name, "dnnl_eltwise", 5e-4, false,
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
    bool               ok = false;
    std::vector<float> out;
    uint64_t           calls = 0, declined = 0, engaged = 0;
};

// One compute of the arm's graph on the backend. inject != 0 forces a decline on that call of the site.
run_result run_arm(ggml_backend_t backend, const arm & a, int inject) {
    run_result res;
    ggml_sycl_test_scratchpad_sites_reset();
    if (inject != 0 && !ggml_sycl_test_inject_scratchpad_decline(a.site, inject)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", a.name, a.site);
        return res;
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
    if (g.w) {
        ggml_backend_tensor_set(g.w, w.data(), 0, w.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: %s: graph compute returned %d\n", a.name, (int) status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    // Recording would route the softmax and MUL arms around their sites, so a recorded run is void.
    if (ggml_sycl::test_backend_has_exec_graph(backend)) {
        fprintf(stderr, "FAIL: %s: the backend recorded an executable graph; the arm is void (run with GGML_SYCL_DISABLE_GRAPH=1)\n",
                a.name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    res.out.resize((size_t) ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, res.out.data(), 0, res.out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    if (!ggml_sycl_test_scratchpad_site_counts(a.site, &res.calls, &res.declined, &res.engaged)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", a.name, a.site);
        return res;
    }
    ggml_sycl_test_inject_scratchpad_decline(a.site, 0);
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
        if (off.calls != 1 || off.engaged != 1 || off.declined != 0) {
            fprintf(stderr,
                    "FAIL: %s: the undeclined run did not engage the wrapper exactly once (calls=%llu engaged=%llu declined=%llu); "
                    "the arm is VOID -- check GGML_SYCL_ONEDNN_SOFTMAX / GGML_SYCL_ONEDNN_MUL and the shape thresholds\n",
                    a.name, (unsigned long long) off.calls, (unsigned long long) off.engaged,
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

    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "test-sycl-dnnl-decline-fallthrough: FAIL\n");
        return 1;
    }
    printf("test-sycl-dnnl-decline-fallthrough: PASS\n");
    return 0;
}
