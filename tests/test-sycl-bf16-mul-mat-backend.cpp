// GPU test for llama.cpp-9qjy: a BF16 dense weight uploaded to the SYCL backend
// must be admitted by supports_op and computed natively from the BF16 bytes --
// through the real dispatch (ggml_sycl_mul_mat -> ggml_sycl_mul_mat_bf16_weight),
// not just the kernel header test-sycl-bf16-mul-mat covers.
//
// The weight lives in a WEIGHTS-usage SYCL buffer under a real tensor name, as a
// model weight does. One graph per activation width exercises the executor's
// paths: N = 1 and 3 take the skinny kernel; N = 40 takes the oneDNN route
// (a bf16 copy of the activations and a bf16 x bf16 -> f32 GEMM) where the build
// has oneDNN and GGML_SYCL_ONEDNN_PP is not 0, and the tiled kernel otherwise.
// References: double-precision sums over the BF16-decoded weights and the F32
// activations, and over the same activations rounded to bf16, which is what the
// CPU backend computes for a BF16 weight (its vec_dot_type is BF16). Every
// tolerance scales with sum |w * x|. The native kernels accumulate the F32
// activations in F32: 1e-4 against the F32 reference. The route reads the bf16
// copy: 2e-4 against the bf16 reference, and closer to it than to the F32 one,
// the witness that the route ran (test-sycl-bf16-onednn-mul-mat-device counts
// its GEMM submissions).
//
// Before 9qjy this graph aborted or was declined whenever the planner had filled
// VRAM, because the executor lazily allocated an unplanned F32 copy of the weight.
//
// No model is loaded: a ~1 MB synthetic weight, a few small graphs. Pinned to
// level_zero:1 by the registration like the other small GPU tests.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "mul-mat-bf16.hpp"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static constexpr int64_t K = 2048;
static constexpr int64_t M = 256;

static constexpr double TOL_NATIVE = 1e-4;  // the native kernels against double over F32 x
static constexpr double TOL_ROUTE  = 2e-4;  // the oneDNN route against double over bf16(x)

// The oneDNN PP admission the route takes (ggml_sycl_mul_mat_bf16_onednn_takes) and the planner marks under; declared
// in common.hpp and exported by libggml-sycl, so this test asks the backend's own answer rather than re-reading its
// environment.
bool ggml_sycl_onednn_pp_type_admitted(ggml_type type);

// Whether the oneDNN route computes an op of n_cols columns: a oneDNN build, the oneDNN PP admission for BF16, and more
// columns than the skinny kernel keeps.
static bool route_expected(int64_t n_cols) {
#if GGML_SYCL_DNNL
    return n_cols > ggml_sycl_bf16::SKINNY_MAX_COLS && ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16);
#else
    (void) n_cols;
    return false;
#endif
}

static bool run_width(ggml_backend_t             backend,
                      ggml_backend_t             cpu,
                      ggml_tensor *              weight,
                      const std::vector<float> & w_dec,
                      int64_t                    n_cols) {
    std::mt19937                          rng(77 + static_cast<unsigned>(n_cols));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t         max_nodes = 8;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);

    ggml_init_params iparams = { mem_size, mem_buffer.data(), /*no_alloc=*/true };
    ggml_context *   ctx     = ggml_init(iparams);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return false;
    }
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_cols);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * out = ggml_mul_mat(ctx, weight, x);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);

    // The op must be claimed by the SYCL backend: a decline would send it to the CPU
    // backend below and this test would pass without touching the executor.
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "FAIL: SYCL supports_op declined BF16 weight x F32 MUL_MAT (n_cols=%lld)\n",
                     (long long) n_cols);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_t       backends[2] = { backend, cpu };
    ggml_backend_sched_t sched       = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
    if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "FAIL: scheduler allocation (n_cols=%lld)\n", (long long) n_cols);
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched, out) != backend) {
        std::fprintf(stderr, "FAIL: MUL_MAT was scheduled on %s, not the SYCL backend\n",
                     ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, out)));
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> x_host(static_cast<size_t>(K * n_cols));
    for (float & v : x_host) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute (n_cols=%lld)\n", (long long) n_cols);
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> y(static_cast<size_t>(M * n_cols));
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));

    // The route is held to the reference over bf16(x), the native kernels to the one over F32 x.
    const bool   route   = route_expected(n_cols);
    const double tol     = route ? TOL_ROUTE : TOL_NATIVE;
    int          bad     = 0;
    double       err_f32 = 0.0;  // max err / sum|w*x| against double over F32 x
    double       err_b16 = 0.0;  // max err / sum|w*x| against double over bf16(x)
    for (int64_t n = 0; n < n_cols; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            double ref_f = 0.0;
            double ref_b = 0.0;
            double mag   = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                const double w = w_dec[m * K + k];
                const double x = x_host[n * K + k];
                ref_f += w * x;
                ref_b += w * ggml_bf16_to_fp32(ggml_fp32_to_bf16(x_host[n * K + k]));
                mag += std::fabs(w * x);
            }
            mag                = mag > 0.0 ? mag : 1.0;
            const double got   = y[n * M + m];
            const double rel_f = std::fabs(got - ref_f) / mag;
            const double rel_b = std::fabs(got - ref_b) / mag;
            err_f32            = std::fmax(err_f32, rel_f);
            err_b16            = std::fmax(err_b16, rel_b);
            if (!((route ? rel_b : rel_f) <= tol)) {
                if (bad < 3) {
                    std::fprintf(stderr, "  m=%lld n=%lld got=%.9g ref(f32 x)=%.9g ref(bf16 x)=%.9g\n", (long long) m,
                                 (long long) n, got, ref_f, ref_b);
                }
                ++bad;
            }
        }
    }
    // A route that read the F32 activations would be no closer to the reference over bf16(x).
    const bool signature = !route || err_b16 < err_f32;
    std::printf("n_cols=%-3lld %s max err/S: vs ref(f32 x) %.3g, vs ref(bf16 x) %.3g (bound %.0e on the %s) %s\n",
                (long long) n_cols, route ? "route " : "native", err_f32, err_b16, tol, route ? "bf16 one" : "f32 one",
                bad == 0 && signature ? "ok" : "BAD");
    if (!signature) {
        std::fprintf(stderr, "FAIL: n_cols=%lld: the route is no closer to the reference over bf16(x)\n",
                     (long long) n_cols);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return bad == 0 && signature;
}

int main(int, char ** argv) {
    // libccl memoizes the selector before main() runs, so a plain setenv() does not work
    // (llama.cpp-2x3m); see tests/sycl-selector-fallback.hpp.
    sycl_test_selector_fallback(argv, "level_zero:1");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::fprintf(stderr,
                     "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                     "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params wparams = { ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    ggml_context *   wctx    = ggml_init(wparams);
    ggml_tensor *    weight  = ggml_new_tensor_2d(wctx, GGML_TYPE_BF16, K, M);
    // A real tensor name: the weight predicate requires one, as a loaded model's weights have.
    ggml_set_name(weight, "blk.0.ffn_gate_inp.weight");

    ggml_backend_buffer_t weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, ggml_backend_sycl_buffer_type(0));
    if (!weight_buf) {
        std::fprintf(stderr, "FAIL: failed to allocate the weight buffer\n");
        ggml_free(wctx);
        ggml_backend_free(cpu);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937                          rng(2026);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ggml_bf16_t>              w_bits(static_cast<size_t>(K * M));
    std::vector<float>                    w_dec(w_bits.size());
    for (size_t i = 0; i < w_bits.size(); ++i) {
        w_bits[i] = ggml_fp32_to_bf16(dist(rng));
        w_dec[i]  = ggml_bf16_to_fp32(w_bits[i]);
    }
    ggml_backend_tensor_set(weight, w_bits.data(), 0, ggml_nbytes(weight));

    bool ok = true;
    for (int64_t n : { 1, 3, 40 }) {
        ok = run_width(backend, cpu, weight, w_dec, n) && ok;
    }

    ggml_backend_buffer_free(weight_buf);
    ggml_free(wctx);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    if (!ok) {
        std::fprintf(stderr, "test-sycl-bf16-mul-mat-backend: FAIL\n");
        return 1;
    }
    std::printf("test-sycl-bf16-mul-mat-backend: PASS\n");
    return 0;
}
