// tests/test-dmmv-q6k-coalesced.cpp
// Test that Q6_K DMMV with coalesced layout produces same output as CPU backend.
//
// Build: cmake --build build --target test-dmmv-q6k-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-dmmv-q6k-coalesced

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"
#include "ggml-sycl.h"
#include "ggml.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Coalesced tile configuration (must match common.hpp)
#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif
constexpr int WARP_SIZE = GGML_SYCL_WARP_SIZE;
constexpr int MMVQ_COALESCED_TILE_BLOCKS = WARP_SIZE;

// Mirrors tile_count()/the decomposition loops in dmmv.cpp: largest power-of-2
// tile first, capped at 32 blocks. One warp per tile, so this is also the
// work-group width the variable-tile kernel launches with.
static int q6k_tile_count(int blocks_per_row) {
    int count     = 0;
    int remaining = blocks_per_row;
    while (remaining > 0) {
        int ts = 1;
        while (ts * 2 <= remaining && ts < MMVQ_COALESCED_TILE_BLOCKS) {
            ts *= 2;
        }
        remaining -= ts;
        count++;
    }
    return count;
}

// Independent f64 reference: dequantize Q6_K exactly as ggml-quants.c does and
// dot with the raw f32 activations. Neither backend is consulted, so it can say
// which of the two is the outlier when they disagree (the GPU consumes f32
// activations; the CPU backend quantizes them to Q8_K first, and that modelling
// difference -- not GPU error -- is what the tolerances below absorb).
static double q6k_exact_dot(const block_q6_K * blocks, const float * y, int blocks_per_row) {
    double acc = 0.0;
    for (int b = 0; b < blocks_per_row; ++b) {
        const block_q6_K & blk = blocks[b];
        const double       d   = ggml_fp16_to_fp32(blk.d);
        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t * ql = blk.ql + (n / 128) * 64;
            const uint8_t * qh = blk.qh + (n / 128) * 32;
            const int8_t *  sc = blk.scales + (n / 128) * 8;
            const float *   yy = y + b * QK_K + n;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int8_t) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int8_t) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * sc[is + 0] * q1 * (double) yy[l];
                acc += d * sc[is + 2] * q2 * (double) yy[l + 32];
                acc += d * sc[is + 4] * q3 * (double) yy[l + 64];
                acc += d * sc[is + 6] * q4 * (double) yy[l + 96];
            }
        }
    }
    return acc;
}

// Watches the ggml log for the two warnings that mean the coalesced DMMV kernel
// did NOT run. Both are GGML_LOG_WARN, so they reach a ggml_log_set callback
// without any verbosity plumbing.
//
// This is only half the gate, and the weaker half: silence here means "nothing
// complained", which is also what a run that dispatched some entirely different
// kernel produces, since only a refusal or a fallback says anything at all.
// run_mul_mat_backend() therefore pairs it with a positive assertion that the
// weight really is in the coalesced layout, and evaluates both.
struct dispatch_probe {
    bool saw_layout_refusal = false;
    bool saw_blas_fallback  = false;

    void reset() {
        saw_layout_refusal = false;
        saw_blas_fallback  = false;
    }
};

static dispatch_probe g_probe;

static void dispatch_probe_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) {
        return;
    }
    if (strstr(text, "not eligible for tensor layout")) {
        g_probe.saw_layout_refusal = true;
    }
    if (strstr(text, "Generic BLAS fallback")) {
        g_probe.saw_blas_fallback = true;
    }
    // Forward everything. Installing a callback replaces ggml's default sink, so
    // filtering here would silently drop the device banner and load lines that
    // make a gate log readable -- observing the stream must not consume it.
    GGML_UNUSED(level);
    fputs(text, stderr);
}

// Positive control for the probe, run before any case depends on it: a matcher
// that cannot match returns "clean" forever and is indistinguishable from a
// dispatch that behaved. Feeds the two literals through and requires both to
// fire.
//
// Limit worth stating plainly: this proves the MATCHER works, not that the
// backend still emits that text -- it scores my copy of the string against
// itself. If ggml-sycl.cpp:53020 or :57102 is reworded, this stays green while
// the negative half goes blind. That is precisely why the gate also has a
// positive half (the layout-mode assertion), which depends on no log text.
static bool dispatch_probe_self_test() {
    g_probe.reset();
    dispatch_probe_log(GGML_LOG_LEVEL_WARN, "[SYCL] Forced kernel DMMV_COALESCED not eligible for tensor layout\n",
                       nullptr);
    dispatch_probe_log(GGML_LOG_LEVEL_WARN, "[MUL_MAT] Generic BLAS fallback for weight (type=14 batch=1)\n", nullptr);
    const bool ok = g_probe.saw_layout_refusal && g_probe.saw_blas_fallback;
    g_probe.reset();
    printf("  probe self-test: %s\n", ok ? "both refusal strings match" : "FAILED -- the dispatch gate is blind");
    return ok;
}

static bool run_mul_mat_backend(ggml_backend_t backend, ggml_type weight_type,
                                const void * weight_data, size_t weight_size,
                                const float * input_data, int n_embd, int n_rows,
                                std::vector<float> & output, bool require_coalesced = false) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    struct ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    // The name is load-bearing, not cosmetic. The COALESCED layout is chosen at
    // ggml_backend_tensor_set time by layout_policy::get_optimal(), keyed on
    // infer_tensor_usage(name) (common.hpp). A bare "weight" infers UNKNOWN,
    // which falls through to GGML_LAYOUT_SOA -- and a SoA tensor makes the
    // forced DMMV_COALESCED kernel ineligible, so dispatch silently drops to the
    // generic BLAS fallback and this test measures something else entirely
    // (llama.cpp-ug4p). An attn_* name infers ATTENTION_WEIGHT, which is the
    // production path to COALESCED for Q6_K.
    //
    // Note ggml_sycl::test_layout_override_guard does NOT cover this: it binds
    // in the dispatch-time selectors, while the materialization above consults
    // layout_policy::get_with_override(), whose "override" is the unrelated
    // GGML_SYCL_UNIFIED_* env knob.
    ggml_set_name(weight, "blk.0.attn_q.weight");

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_name(input, "input");

    struct ggml_tensor * out = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(out, "output");

    size_t weight_buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t weight_buffer = ggml_backend_buft_alloc_buffer(buft, weight_buf_size);
    if (!weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(weight_buffer, weight, (void *)ggml_backend_buffer_get_base(weight_buffer));

    size_t input_size = ggml_backend_buft_get_alloc_size(buft, input);
    size_t output_size = ggml_backend_buft_get_alloc_size(buft, out);
    ggml_backend_buffer_t compute_buffer = ggml_backend_buft_alloc_buffer(buft, input_size + output_size + 4096);
    if (!compute_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint8_t * base = (uint8_t *)ggml_backend_buffer_get_base(compute_buffer);
    ggml_backend_tensor_alloc(compute_buffer, input, base);
    ggml_backend_tensor_alloc(compute_buffer, out, base + input_size);

    ggml_backend_tensor_set(weight, weight_data, 0, weight_size);
    ggml_backend_tensor_set(input, input_data, 0, n_embd * sizeof(float));

    // Positive half of the kernel-identity gate: the weight must actually be in
    // the coalesced layout after upload. Without this the test can only observe
    // that nothing complained, which is what let a BLAS-fallback run pass as a
    // coalesced-kernel run.
    //
    // The result is RECORDED, not acted on. Returning here would short-circuit
    // the negative half and make the two halves untestable in one run: the
    // refusal and fallback warnings are only emitted if the graph actually
    // runs, so bailing before compute leaves the probe with nothing to observe.
    // A mutation that breaks the layout would then look like it exercised only
    // the layout assertion, when in fact it drives both.
    enum ggml_layout_mode weight_layout = GGML_LAYOUT_AOS;
    bool                  layout_ok     = true;
    if (require_coalesced) {
        weight_layout = weight->layout ? weight->layout->mode : GGML_LAYOUT_AOS;
        layout_ok     = (weight_layout == GGML_LAYOUT_COALESCED);
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    if (require_coalesced) {
        g_probe.reset();
    }
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool success = (status == GGML_STATUS_SUCCESS);

    // Negative half: dispatch must not have declined the kernel or fallen back.
    if (success && require_coalesced) {
        const bool dispatch_ok = !(g_probe.saw_layout_refusal || g_probe.saw_blas_fallback);
        if (!layout_ok) {
            printf("  FAIL: weight layout is %d, expected GGML_LAYOUT_COALESCED (%d) -- the coalesced "
                   "kernel cannot run\n",
                   (int) weight_layout, (int) GGML_LAYOUT_COALESCED);
        }
        if (!dispatch_ok) {
            printf("  FAIL: dispatch declined the coalesced DMMV kernel (layout_refusal=%d blas_fallback=%d)\n",
                   (int) g_probe.saw_layout_refusal, (int) g_probe.saw_blas_fallback);
        }
        if (!layout_ok || !dispatch_ok) {
            printf("  kernel-identity gate: layout_ok=%d dispatch_ok=%d -- any numbers below would come "
                   "from another path\n",
                   (int) layout_ok, (int) dispatch_ok);
            success = false;
        }
    }
    if (success) {
        output.resize(n_rows);
        ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(ctx);

    return success;
}

static bool run_dmmv_q6k_coalesced_case(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
                                       int ncols, int nrows, bool verbose) {
    if (ncols % QK_K != 0) {
        printf("  SKIP: ncols not multiple of QK_K\n");
        return true;
    }
    const int blocks_per_row = ncols / QK_K;
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("  SKIP: blocks_per_row=%d not multiple of %d (coalesced tile)\n",
               blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;
    }

    std::mt19937 rng(2027 + ncols + nrows);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float((size_t)nrows * ncols);
    for (float & v : weight_float) v = dist(rng);

    std::vector<block_q6_K> weight_quant((size_t)nrows * blocks_per_row);
    for (int row = 0; row < nrows; ++row) {
        quantize_row_q6_K_ref(weight_float.data() + (size_t)row * ncols,
                              weight_quant.data() + (size_t)row * blocks_per_row,
                              ncols);
    }

    std::vector<float> input_data(ncols);
    for (float & v : input_data) v = dist(rng);

    const size_t weight_bytes = weight_quant.size() * sizeof(block_q6_K);

    std::vector<float> gpu_out;
    std::vector<float> cpu_out;
    if (!run_mul_mat_backend(gpu_backend, GGML_TYPE_Q6_K, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, gpu_out, /*require_coalesced=*/true)) {
        printf("  FAIL: GPU backend compute failed\n");
        return false;
    }
    if (!run_mul_mat_backend(cpu_backend, GGML_TYPE_Q6_K, weight_quant.data(), weight_bytes,
                             input_data.data(), ncols, nrows, cpu_out)) {
        printf("  FAIL: CPU backend compute failed\n");
        return false;
    }

    // The two backends do not model the activations the same way: the GPU
    // kernel consumes the raw f32 vector, the CPU backend quantizes it to Q8_K
    // first. The residual is a random walk over ncols terms, so it grows as
    // sqrt(ncols) -- measured against q6k_exact_dot on the host, the CPU side
    // carries essentially all of it (llama.cpp-ug4p):
    //
    //   ncols   worst |cpu - exact|   worst |kernel - exact|
    //    8192   0.35                  8e-6
    //   16384   0.42                  9e-6
    //   32768   0.73                  4e-6
    //
    // A fixed 0.5 therefore fails a *correct* kernel at 32768 columns. Scale it
    // with sqrt(ncols) so every shape gets the same ~4 sigma of headroom, and
    // anchor the constant at the 8192 value so the long-standing gate there is
    // unchanged.
    const float abs_tol = 0.5f * sqrtf((float) ncols / 8192.0f);
    const float rel_tol = 0.2f;
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int errors = 0;
    std::vector<int> failing_rows;
    for (int i = 0; i < nrows; ++i) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        float denom = fmaxf(1.0f, fabsf(cpu_out[i]));
        float rel = diff / denom;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > abs_tol && rel > rel_tol) {
            errors++;
            failing_rows.push_back(i);
        }
    }

    const bool pass = (errors == 0);
    if (verbose || !pass) {
        printf("  Q6_K coalesced ncols=%d nrows=%d blocks_per_row=%d num_tiles=%d: "
               "errors=%d max_diff=%.6f max_rel=%.6f %s\n",
               ncols, nrows, blocks_per_row, q6k_tile_count(blocks_per_row),
               errors, max_diff, max_rel, pass ? "PASS" : "FAIL");
    }

    if (pass) {
        return true;
    }

    // A failure here is worth one full diagnosis: the kernel arithmetic and the
    // coalesced layout indexing were both machine-checked against q6k_exact_dot
    // on the host (llama.cpp-ug4p) and reproduce the exact answer to ~1e-5 at
    // every shape, so a gross on-device disagreement is a data-path or
    // execution defect and needs to be classified, not just counted.
    printf("  --- diagnosis (llama.cpp-ug4p) ---\n");
    for (int i : failing_rows) {
        const double exact = q6k_exact_dot(weight_quant.data() + (size_t) i * blocks_per_row,
                                           input_data.data(), blocks_per_row);
        printf("    row %d: gpu=%.6f cpu=%.6f exact_f64=%.6f | gpu-exact=%.6f cpu-exact=%.6f%s\n",
               i, gpu_out[i], cpu_out[i], exact,
               gpu_out[i] - exact, cpu_out[i] - exact,
               std::isfinite(gpu_out[i]) ? "" : "  [gpu value is NOT FINITE]");
        if (gpu_out[i] == 0.0f) {
            printf("      gpu value is exactly 0 -- output slot likely never written\n");
        }
        // Index shear: does this row's GPU answer belong to a different row?
        for (int j = 0; j < nrows; ++j) {
            if (j != i && fabsf(gpu_out[i] - cpu_out[j]) < 1e-2f) {
                printf("      gpu[%d] matches cpu[%d] -- row index shear\n", i, j);
                break;
            }
        }
    }

    // Deterministic or racy? Re-run the same shape on the GPU with identical
    // inputs; a bit-identical repeat rules out a race in the materialization.
    std::vector<float> gpu_out2;
    if (run_mul_mat_backend(gpu_backend, GGML_TYPE_Q6_K, weight_quant.data(), weight_bytes,
                            input_data.data(), ncols, nrows, gpu_out2, /*require_coalesced=*/true)) {
        // The verdict is about the DEFECT, so it is scored on the failing rows
        // only. Scoring it on all-rows bit-equality made it announce "racy"
        // while its own detail lines showed the failing row identical in both
        // runs: any last-bit jitter in a healthy row -- ordinary for a threaded
        // or tree reduction -- flipped the verdict. Whole-buffer drift is still
        // reported, as context rather than as the verdict.
        bool defect_stable = true;
        for (int i : failing_rows) {
            if (gpu_out[i] != gpu_out2[i]) {
                defect_stable = false;
            }
            printf("      row %d: run1=%.6f run2=%.6f\n", i, gpu_out[i], gpu_out2[i]);
        }
        printf("    repeat run: failing rows are %s\n",
               defect_stable ? "bit-identical (deterministic defect)"
                             : "DIFFERENT (racy / order-dependent defect)");

        int   drifted  = 0;
        float max_drift = 0.0f;
        for (int i = 0; i < nrows; ++i) {
            const float d = fabsf(gpu_out[i] - gpu_out2[i]);
            if (d != 0.0f) {
                drifted++;
            }
            if (d > max_drift) {
                max_drift = d;
            }
        }
        printf("    run-to-run drift across all rows: %d/%d rows differ, max=%.9f\n", drifted, nrows, max_drift);
    }
    return false;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    printf("=== Q6_K DMMV Coalesced Tests ===\n");

    ggml_log_set(dispatch_probe_log, nullptr);
    if (!dispatch_probe_self_test()) {
        printf("=== Summary: FAIL ===\n");
        return 1;
    }

    ggml_sycl::test_layout_override_guard guard(GGML_LAYOUT_COALESCED);
    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);
    // The layout override and GGML_SYCL_FORCE_DMMV both bind DOWNSTREAM of the
    // batch=1 TG fast-path (ggml-sycl.cpp, "TG fast-path: batch=1, single-device,
    // quantized -> MMVQ"), which returns before either is consulted. Q6_K is in
    // ggml_sycl_supports_reorder_mmvq, and every case below has src1->ne[1] == 1,
    // so without this line every dispatch here goes to MMVQ_COALESCED and no
    // dmmv.cpp code runs at all -- the same miss llama.cpp-szv8 spent three rounds
    // on. GGML_SYCL_TG_FAST=0 is that path's documented disable.
    setenv("GGML_SYCL_TG_FAST", "0", 1);

    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("SKIP: Could not initialize SYCL backend\n");
        return 0;
    }
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("SKIP: Could not initialize CPU backend\n");
        ggml_backend_free(gpu_backend);
        return 0;
    }

    bool ok = true;
    // Single-tile shapes (blocks_per_row == 32): one warp per row, so the
    // multi-tile offset arithmetic in the kernel is never exercised.
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 1, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 8, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 32, true);
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 8192, 128, true);
    // Multi-tile shapes: these are the only cases that reach the per-warp tile
    // offset loops and the cross-warp shared-memory reduction. 16384x8 used to
    // be the sole one, which left the tile-stride arithmetic uncovered at every
    // other row count (llama.cpp-ug4p).
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 16384, 8, true);    // 2 tiles
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 16384, 128, true);  // 2 tiles, wide
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 24576, 8, true);    // 3 tiles
    ok &= run_dmmv_q6k_coalesced_case(gpu_backend, cpu_backend, 32768, 32, true);   // 4 tiles

    ggml_backend_free(cpu_backend);
    ggml_backend_free(gpu_backend);

    printf("=== Summary: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
