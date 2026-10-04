// GPU test for llama.cpp-9qjy: a BF16 dense weight uploaded to the SYCL backend
// must be admitted by supports_op and computed natively from the BF16 bytes --
// through the real dispatch (ggml_sycl_mul_mat -> ggml_sycl_mul_mat_bf16_weight),
// not just the kernel header test-sycl-bf16-mul-mat covers.
//
// The weight lives in a WEIGHTS-usage SYCL buffer under a real tensor name, as a
// model weight does. One graph per activation width exercises both executor
// kernels (N = 1 and 3 take the skinny kernel, N = 40 the tiled one). Reference:
// double-precision sum over the BF16-decoded weights and the F32 activations;
// the executor accumulates in F32, so the tolerance scales with sum |w * x|.
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
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static constexpr int64_t K = 2048;
static constexpr int64_t M = 256;

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

    int    bad     = 0;
    double max_err = 0.0;
    for (int64_t n = 0; n < n_cols; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            double ref = 0.0;
            double mag = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                const double p = static_cast<double>(w_dec[m * K + k]) * x_host[n * K + k];
                ref += p;
                mag += std::fabs(p);
            }
            const double err = std::fabs(y[n * M + m] - ref);
            max_err          = std::fmax(max_err, mag > 0.0 ? err / mag : err);
            if (!(err <= 1e-4 * mag + 1e-30)) {
                if (bad < 3) {
                    std::fprintf(stderr, "  m=%lld n=%lld got=%.9g ref=%.9g\n", (long long) m, (long long) n,
                                 (double) y[n * M + m], ref);
                }
                ++bad;
            }
        }
    }
    std::printf("n_cols=%-3lld max_rel_err=%.3g %s\n", (long long) n_cols, max_err, bad == 0 ? "ok" : "BAD");

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return bad == 0;
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
