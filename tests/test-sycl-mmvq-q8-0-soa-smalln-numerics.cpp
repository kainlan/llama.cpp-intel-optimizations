// GPU numerics test for the Q8_0 SOA MMVQ small-N occupancy geometry
// (llama.cpp-6cgq): reorder_mul_mat_vec_q8_0_q8_1_sycl's small-N branch
// (mmvq.cpp, gated by GGML_SYCL_MMVQ_Q8_SOA_SMALLN, default ON) must produce
// the SAME output as the unmodified fixed-16 launch it replaces for small N,
// and the SOA kernel's output (either geometry) must agree with both a CPU
// reference and the COALESCED kernel's output for the same weights and
// input.
//
// This is intentionally NOT run by the implementer (no GPU access in this
// role, per the fork's Hard-Won Rules); it is written for `main` to build
// and run. Run twice -- GGML_SYCL_MMVQ_Q8_SOA_SMALLN unset (default ON) and
// =0 -- since the env accessor caches its value once per process
// (mmvq_q8_0_soa_smalln_enabled() in mmvq.cpp), so one process cannot
// exercise both branches; that is why this binary does not setenv it itself.
//
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-mmvq-q8-0-soa-smalln-numerics
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_MMVQ_Q8_SOA_SMALLN=0 \
//       ./build/bin/test-sycl-mmvq-q8-0-soa-smalln-numerics
//
// Strategy per shape: build ONE weight tensor's worth of random Q8_0 blocks
// and ONE input tensor's worth of random F32 activations, materialize the
// SAME bytes as two SEPARATE device weight tensors -- one forced SOA, one
// forced COALESCED, via the ggml-sycl-private-fixtures layout-override hook
// (the same mechanism tests/test-q8-0-layout-cache-path-mmvq.cpp uses to
// force SOA caching) -- run MUL_MAT on each, and compare:
//   (a) SOA GPU output vs a CPU dequant+dot reference,
//   (b) COALESCED GPU output vs the same CPU reference,
//   (c) SOA GPU output vs COALESCED GPU output directly.
// (a)+(b) catch either kernel drifting from correct; (c) is the direct
// "SOA vs COALESCED" comparison the ticket asked for, and is the tightest of
// the three since both read literally the same weight bytes.

#include "common.hpp"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml.h"
#include "test-skip.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define QK8_0 32

typedef struct {
    ggml_fp16_t d;
    int8_t      qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

static int g_failures = 0;

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft,
                                                 ggml_tensor *              tensor,
                                                 ggml_backend_buffer_usage  usage) {
    const size_t          size   = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static void fill_q8_0_random(block_q8_0_test * blocks, int nblocks, std::mt19937 & rng) {
    std::uniform_int_distribution<int>    qdist(-120, 120);
    std::uniform_real_distribution<float> ddist(0.001f, 0.05f);
    for (int i = 0; i < nblocks; ++i) {
        blocks[i].d = ggml_fp32_to_fp16(ddist(rng));
        for (int j = 0; j < QK8_0; ++j) {
            blocks[i].qs[j] = static_cast<int8_t>(qdist(rng));
        }
    }
}

static void compute_reference(const block_q8_0_test * weights,
                              const float *           input,
                              float *                 output,
                              int                     ncols,
                              int                     nrows,
                              int                     batch) {
    const int blocks_per_row = ncols / QK8_0;
    for (int b = 0; b < batch; ++b) {
        const float * input_col = input + (size_t) b * ncols;
        float *       out_col   = output + (size_t) b * nrows;
        for (int row = 0; row < nrows; ++row) {
            double                  sum        = 0.0;
            const block_q8_0_test * row_blocks = weights + (size_t) row * blocks_per_row;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                const block_q8_0_test * w = row_blocks + blk;
                const float             d = ggml_fp16_to_fp32(w->d);
                const float *           x = input_col + (size_t) blk * QK8_0;
                for (int j = 0; j < QK8_0; ++j) {
                    sum += (double) d * (double) w->qs[j] * (double) x[j];
                }
            }
            out_col[row] = (float) sum;
        }
    }
}

// Runs one MUL_MAT under the given forced layout and returns the GPU output
// (empty on failure, with a printed reason). weight_data/input_data are the
// SAME bytes for every call in a shape so SOA and COALESCED read identical
// weights.
static std::vector<float> run_layout(ggml_backend_t                       backend,
                                     ggml_layout_mode                     layout,
                                     const char *                         layout_name,
                                     int                                  ncols,
                                     int                                  nrows,
                                     int                                  batch,
                                     const std::vector<block_q8_0_test> & weight_data,
                                     const std::vector<float> &           input_data) {
    std::vector<float> empty;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::printf("FAIL [%s]: ggml_init failed\n", layout_name);
        ++g_failures;
        return empty;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    ggml_set_name(weight, "mmvq_q8_0_soa_smalln_weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    ggml_set_name(input, "mmvq_q8_0_soa_smalln_input");
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);

    ggml_backend_buffer_type_t weight_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft    = ggml_backend_get_default_buffer_type(backend);

    ggml_backend_buffer_t weight_buf = alloc_tensor_buffer(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t input_buf  = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t output_buf = alloc_tensor_buffer(dev_buft, output, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    auto cleanup = [&](ggml_sycl_model_token * model) {
        if (weight_buf) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (input_buf) {
            ggml_backend_buffer_free(input_buf);
        }
        if (output_buf) {
            ggml_backend_buffer_free(output_buf);
        }
        ggml_free(ctx);
        if (model) {
            (void) ggml_backend_sycl_model_unloaded_token(*model);
        }
    };

    if (!weight_buf || !input_buf || !output_buf) {
        std::printf("FAIL [%s]: buffer allocation failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    // See test-q8-0-layout-cache-path-mmvq.cpp: host-weight registration is
    // only honoured inside a model-load transaction, and that transaction is
    // the only producer of a dense-weight cache entry -- so registration and
    // upload must both happen inside it (llama.cpp-43uy).
    ggml_sycl::test_clear_host_weight_registry();
    ggml_sycl::test_layout_override_guard layout_guard(layout);

    ggml_sycl_load_txn    load{};
    ggml_sycl_model_token model{};
    if (ggml_backend_sycl_model_load_begin(&load) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL [%s]: model_load_begin failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(block_q8_0_test));

    if (ggml_backend_sycl_model_load_end(load, true, &model) != GGML_SYCL_LIFECYCLE_OK) {
        std::printf("FAIL [%s]: model_load_end failed\n", layout_name);
        ++g_failures;
        cleanup(nullptr);
        return empty;
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    auto resolved = ggml_sycl_resolve(weight, 0);
    if (!resolved || resolved.layout != layout) {
        std::printf("FAIL [%s]: expected layout not resolved (got %d)\n", layout_name,
                    resolved ? (int) resolved.layout : -1);
        ++g_failures;
        cleanup(&model);
        return empty;
    }

    // Overwrite the AoS staging bytes so a kernel that (incorrectly) reads
    // them instead of the resolved/cached layout cannot pass by accident.
    std::memset(weight->data, 0, ggml_nbytes(weight));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::printf("FAIL [%s]: graph compute failed (status=%d)\n", layout_name, (int) status);
        ++g_failures;
        cleanup(&model);
        return empty;
    }

    std::vector<float> gpu_output((size_t) nrows * batch, 0.0f);
    ggml_backend_tensor_get(output, gpu_output.data(), 0, gpu_output.size() * sizeof(float));

    cleanup(&model);
    return gpu_output;
}

static void compare(const char *               what,
                    const std::vector<float> & got,
                    const std::vector<float> & ref,
                    float                      rel_tol,
                    float                      abs_tol) {
    if (got.empty() || ref.empty() || got.size() != ref.size()) {
        std::printf("FAIL [%s]: size mismatch or missing output (got=%zu ref=%zu)\n", what, got.size(), ref.size());
        ++g_failures;
        return;
    }
    float max_diff = 0.0f;
    float max_rel  = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        max_diff         = std::max(max_diff, diff);
        const float rel  = std::fabs(ref[i]) > 1e-6f ? diff / std::fabs(ref[i]) : diff;
        max_rel          = std::max(max_rel, rel);
    }
    const bool ok = max_rel < rel_tol || max_diff < abs_tol;
    std::printf("%s [%s]: max_diff=%.6e max_rel=%.6e (tol rel=%.1e abs=%.1e)\n", ok ? "OK" : "FAIL", what, max_diff,
                max_rel, rel_tol, abs_tol);
    if (!ok) {
        ++g_failures;
    }
}

static void run_shape(ggml_backend_t backend, const char * label, int ncols, int nrows, int batch) {
    std::printf("== shape %s: ncols=%d nrows=%d batch=%d ==\n", label, ncols, nrows, batch);
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks        = nrows * blocks_per_row;

    std::mt19937                 rng(0x6c9au ^ (ncols * 2654435761u) ^ (nrows * 40503u) ^ (unsigned) batch);
    std::vector<block_q8_0_test> weight_data(nblocks);
    fill_q8_0_random(weight_data.data(), nblocks, rng);

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float>                    input_data((size_t) ncols * batch);
    for (float & v : input_data) {
        v = xdist(rng);
    }

    std::vector<float> ref((size_t) nrows * batch);
    compute_reference(weight_data.data(), input_data.data(), ref.data(), ncols, nrows, batch);

    const std::vector<float> soa =
        run_layout(backend, GGML_LAYOUT_SOA, "soa", ncols, nrows, batch, weight_data, input_data);
    const std::vector<float> coalesced =
        run_layout(backend, GGML_LAYOUT_COALESCED, "coalesced", ncols, nrows, batch, weight_data, input_data);

    // Q8_0 x Q8_0 accumulation: generous but not vacuous tolerance, matching
    // test-q8-0-layout-cache-path-mmvq.cpp's precedent for this same op.
    const float rel_tol = 1e-3f;
    const float abs_tol = 5e-2f;
    compare("soa-vs-cpu-reference", soa, ref, rel_tol, abs_tol);
    compare("coalesced-vs-cpu-reference", coalesced, ref, rel_tol, abs_tol);
    // Tighter: both kernels read the SAME bytes, so their outputs should
    // agree closely regardless of either's absolute accuracy against the
    // reference.
    compare("soa-vs-coalesced", soa, coalesced, 1e-4f, 1e-2f);
}

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: no SYCL GPU device available\n");
        return LLAMA_TEST_EXIT_SKIP;
    }

    // Shapes from the llama.cpp-6cgq profile table plus the ticket's
    // requested extra coverage (a non-Mistral K, and a tiny row count).
    run_shape(backend, "k,v N=1024 K=4096 b1", 4096, 1024, 1);
    run_shape(backend, "q,o N=4096 K=4096 b1", 4096, 4096, 1);
    run_shape(backend, "gate/up N=14336 K=4096 b1", 4096, 14336, 1);
    run_shape(backend, "down N=4096 K=14336 b1", 14336, 4096, 1);
    run_shape(backend, "gemma4-ish N=2048 K=2560 b1", 2560, 2048, 1);
    run_shape(backend, "N=10240 K=2560 b1", 2560, 10240, 1);
    run_shape(backend, "tiny N=37 K=4096 b1", 4096, 37, 1);
    // Batch 2 and 8 for the shape the small-N geometry changes most.
    run_shape(backend, "k,v N=1024 K=4096 b2", 4096, 1024, 2);
    run_shape(backend, "k,v N=1024 K=4096 b8", 4096, 1024, 8);

    ggml_backend_free(backend);

    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: q8_0 SOA MMVQ small-N geometry numerics\n");
    return 0;
}
