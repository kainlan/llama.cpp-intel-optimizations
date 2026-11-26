// GPU test of the BF16 weight's oneDNN route: above SKINNY_MAX_COLS columns a dense BF16 MUL_MAT runs as a bf16 copy
// of its F32 activations (rounded to nearest even, in the planned dense src1 buffer) and one bf16 x bf16 -> f32 oneDNN
// GEMM, where the native executor's tiled kernel computed it without XMX (ggml_sycl_mul_mat_bf16_onednn).
//
// Each case computes one MUL_MAT graph directly on the SYCL backend: the weight in a WEIGHTS-usage SYCL buffer under a
// real tensor name, as a model's is, the activations and the output in a COMPUTE buffer. The PRIVATE_TESTING scratchpad
// seam counts the "dnnl_gemm" site, which every gemm() call that reaches its scratchpad passes once, and is the witness
// of which path ran. The route asks gemm() twice: once with query_only before its conversion, once to run the GEMM.
//
//   taken     the route run must count calls == 2, engaged == 1, declined == 0, and leave one primitive in the cache it
//             cleared first. The case is computed twice more with a decline injected: on the first call (the query),
//             so nothing is written and the native kernel computes the op (calls == 1, declined == 1, engaged == 0);
//             and on the second (the GEMM, after the conversion has written the planned buffer), so the native kernel
//             computes the op on the same inputs (calls == 2, declined == 1, engaged == 0). The two native outputs
//             must be bit-identical, and they are the second oracle;
//   declined  a node of at most SKINNY_MAX_COLS columns (decode, and the widths the skinny kernel keeps) never reaches
//             gemm(): calls == 0. So does every node with the route switched off (GGML_SYCL_ONEDNN_PP=0, run with
//             --expect-opt-out by the second registration).
//
// Two cases run first, on the fresh backend, while the planned src1 buffer is still empty:
//   * the route's demand left out of the graph-entry walk (ggml_sycl::test_set_bf16_onednn_walk_counted(false)): the
//     route must decline (calls == 0) without growing the buffer, count one under-estimate against the buffer's
//     cohort, and the native kernel computes the op;
//   * the route's demand counted, with the walk's grows limited below it (ggml_sycl::test_set_dequant_f16_walk_limit),
//     as a RUNTIME zone that cannot hold the copy: the graph must still compute (the walk caps the buffer and does not
//     refuse the graph for a demand with another path), the route declines with one under-estimate counted, and the
//     native kernel computes the op.
// Three layouts the route's arithmetic reads through strides run on the up projection at 129 columns, taken and with
// both injected declines: src1 a view whose columns are wider than K (ldx > K, the padding NaN, so a read of it shows
// in the output), dst columns wider than M (ldy != M, the padding a sentinel that must survive), and src1 of three
// planes (ne2 > 1).
//
// Tolerances. Every comparison is scaled by S = sum over k of |w[m][k] * x[k][n]| for its element, in double.
//   The route against the double reference over bf16(x): err <= 2e-4. The CPU backend computes a BF16 weight's MUL_MAT
//   on bf16 activations (ggml-cpu.c, the BF16 type traits: vec_dot_type GGML_TYPE_BF16), so a bf16 copy is the CPU
//   reference's own arithmetic; what is left is the route's F32 accumulation.
//   The route's error against that reference is smaller than against the reference over F32 x: the witness that the
//   GEMM read the rounded copy, not the F32 activations.
//   The native kernel against the double reference over F32 x: err <= 1e-4 (its F32 accumulation).
//   The route against the native kernel: err <= 4e-3.
// The reference is computed for a sample of output columns (all rows of each), the first, the last and an even spread
// between: a full double reference of the largest case is 14.5e9 products. Every element of every run is checked
// finite. The measured maxima are printed per case.
//
// The shapes are Inkling 2 Mini's dense BF16 FFN (2304 x 3072 up and gate, 3072 x 2304 down) at T = 9, 129, 512 and
// 2048, the widths the probe timed, and Qwen3.8's BF16 matrices (hc_inject M = 4, ssm_alpha M = 48, hc_down M = 320,
// the shared-expert gate M = 1) at T = 9 and 512: the route's speed was measured on the Inkling shapes only, so the
// Qwen shapes are here for correctness. A primitive oneDNN cannot create for a shape is a decline, not a throw: such a
// case fails on its counts and names the shape.
//
// Touches the GPU with single-op graphs (at most ~75 MB) and loads no model; labelled mem-handle so a sweep leaves it
// to the lead. Usage (the registration supplies the env; a bare run needs it too):
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DISABLE_GRAPH=1 ./build/bin/test-sycl-bf16-onednn-mul-mat-device
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DISABLE_GRAPH=1 ./build/bin/test-sycl-bf16-onednn-mul-mat-device
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ONEDNN_PP=0 \
//       ./build/bin/test-sycl-bf16-onednn-mul-mat-device --expect-opt-out

#include "gemm.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-sycl-test.hpp"
#include "ggml-sycl.h"
#include "ggml.h"
#include "mul-mat-bf16.hpp"
#include "sycl-selector-fallback.hpp"
#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "zone-sizing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if !defined(GGML_SYCL_PRIVATE_TESTING)
#    error "this test requires GGML_SYCL_PRIVATE_TESTING (link against ggml-sycl-private-fixtures)"
#endif

namespace {

constexpr const char * SITE                = "dnnl_gemm";
constexpr double       TOL_ROUTE           = 2e-4;  // the route against double over bf16(x)
constexpr double       TOL_NATIVE          = 1e-4;  // the native kernel against double over F32 x
constexpr double       TOL_ROUTE_VS_NATIVE = 4e-3;
// The planned src1 buffer's cohort: a short buffer counts its under-estimate under this name.
constexpr const char * SRC1_COHORT         = "mul-mat-dequant-f16-src1";
constexpr int64_t      PAD                 = 7;           // extra elements per column in the strided layouts
constexpr float        DST_SENTINEL        = -12345.25f;  // dst padding: no path may write it

// How a case lays out its activations and its output.
enum case_layout {
    CASE_LAYOUT_PLAIN,         // src1 [K, N] and dst [M, N], contiguous
    CASE_LAYOUT_STRIDED_SRC1,  // src1 a view of [K + PAD, N]: ldx = K + PAD, the padding NaN
    CASE_LAYOUT_STRIDED_DST,   // dst columns of M + PAD: ldy = M + PAD, the padding DST_SENTINEL
    CASE_LAYOUT_SRC1_3D,       // src1 [K, N / 3, 3] and dst [M, N / 3, 3]: three planes
};

float input_value(int64_t i, int64_t salt) {
    // a deterministic spread over [-1, 1) with no short period
    const uint64_t h = (uint64_t) (i * 2654435761LL + salt * 40503LL) % 1000003ULL;
    return -1.0f + 2.0f * (float) h / 1000003.0f;
}

float x_value(int64_t K, int64_t k, int64_t n) {
    return input_value(k + K * n, 7);
}

// A BF16 weight of K x M placed once in a WEIGHTS-usage SYCL buffer, with its decoded values for the reference.
struct placed_weight {
    std::string           name;
    int64_t               K = 0, M = 0;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         w   = nullptr;
    std::vector<float>    dec;

    ~placed_weight() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

std::unique_ptr<placed_weight> place_weight(const char * name, int64_t K, int64_t M, int64_t salt) {
    auto             pw      = std::make_unique<placed_weight>();
    ggml_init_params wparams = { ggml_tensor_overhead() + 1024, nullptr, /*no_alloc=*/true };
    pw->name                 = name;
    pw->K                    = K;
    pw->M                    = M;
    pw->ctx                  = ggml_init(wparams);
    if (!pw->ctx) {
        return nullptr;
    }
    pw->w = ggml_new_tensor_2d(pw->ctx, GGML_TYPE_BF16, K, M);
    // A real tensor name: the weight predicate requires one, as a loaded model's weights have.
    ggml_set_name(pw->w, name);
    pw->buf = ggml_backend_alloc_ctx_tensors_from_buft(pw->ctx, ggml_backend_sycl_buffer_type(0));
    if (!pw->buf) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(pw->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<ggml_bf16_t> bits((size_t) (K * M));
    pw->dec.resize(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        bits[i]    = ggml_fp32_to_bf16(input_value((int64_t) i, salt));
        pw->dec[i] = ggml_bf16_to_fp32(bits[i]);
    }
    ggml_backend_tensor_set(pw->w, bits.data(), 0, ggml_nbytes(pw->w));
    return pw;
}

struct run_result {
    bool               ok = false;
    std::vector<float> out;
    uint64_t           calls = 0, declined = 0, engaged = 0;
    size_t             cache_entries = 0, scratchpad_bytes = 0;
};

run_result run_case(ggml_backend_t        backend,
                    const placed_weight & pw,
                    int64_t               N,
                    int                   inject,
                    const std::string &   label,
                    case_layout           layout = CASE_LAYOUT_PLAIN) {
    run_result res;
    ggml_sycl_test_scratchpad_sites_reset();
    if (inject != 0 && !ggml_sycl_test_inject_scratchpad_decline(SITE, inject)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", label.c_str(), SITE);
        return res;
    }
    get_dnnl_primitive_cache().clear();

    const size_t         max_nodes = 8;
    const size_t         mem_size  = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> mem_buffer(mem_size);
    ggml_init_params iparams = { /*.mem_size   =*/mem_size, /*.mem_buffer =*/mem_buffer.data(), /*.no_alloc   =*/true };
    ggml_context *   ctx     = ggml_init(iparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: %s: ggml_init failed\n", label.c_str());
        return res;
    }
    // `x_in` is the tensor the activations are written to: x itself, or the wider tensor a strided x views.
    const int64_t x_cols = layout == CASE_LAYOUT_STRIDED_SRC1 ? pw.K + PAD : pw.K;
    ggml_tensor * x_in   = nullptr;
    ggml_tensor * x      = nullptr;
    if (layout == CASE_LAYOUT_SRC1_3D) {
        x_in = x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pw.K, N / 3, 3);
    } else {
        x_in = x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_cols, N);
        if (layout == CASE_LAYOUT_STRIDED_SRC1) {
            x = ggml_view_2d(ctx, x_in, pw.K, N, x_in->nb[1], 0);
        }
    }
    ggml_set_input(x_in);
    ggml_tensor * out = ggml_mul_mat(ctx, pw.w, x);
    ggml_set_output(out);
    const int64_t ldy = layout == CASE_LAYOUT_STRIDED_DST ? pw.M + PAD : pw.M;
    if (layout == CASE_LAYOUT_STRIDED_DST) {
        // Set before the tensor is allocated, so its allocation holds the padded columns.
        out->nb[1] = (size_t) ldy * sizeof(float);
        out->nb[2] = out->nb[1] * (size_t) out->ne[1];
        out->nb[3] = out->nb[2] * (size_t) out->ne[2];
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    // The executor aborts on an op outside its predicate: ask the admission first, so such a case fails by name.
    if (!ggml_backend_supports_op(backend, out)) {
        fprintf(stderr, "FAIL: %s: SYCL supports_op declined the BF16 weight's MUL_MAT\n", label.c_str());
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_sycl_buffer_type(0));
    if (!buf) {
        fprintf(stderr, "FAIL: %s: could not allocate the graph's tensors on the SYCL device\n", label.c_str());
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> xv((size_t) (x_cols * N), std::nanf(""));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < pw.K; ++k) {
            xv[(size_t) (k + x_cols * n)] = x_value(pw.K, k, n);
        }
    }
    ggml_backend_tensor_set(x_in, xv.data(), 0, xv.size() * sizeof(float));
    std::vector<float> raw(ggml_nbytes(out) / sizeof(float), DST_SENTINEL);
    ggml_backend_tensor_set(out, raw.data(), 0, raw.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: %s: graph compute returned %d\n", label.c_str(), (int) status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    // A recorded graph would make the route decline (it does not create a primitive inside a recording): void.
    if (ggml_sycl::test_backend_has_exec_graph(backend)) {
        fprintf(stderr,
                "FAIL: %s: the backend recorded an executable graph; the case is void (run with "
                "GGML_SYCL_DISABLE_GRAPH=1)\n",
                label.c_str());
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
    ggml_backend_tensor_get(out, raw.data(), 0, raw.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    // Compact the output to [M, N] and check that the padding between dst columns kept the sentinel.
    res.out.resize((size_t) (pw.M * N));
    for (int64_t n = 0; n < N; ++n) {
        // The last column ends at row M: the allocation holds no padding after it.
        const int64_t rows = n + 1 < N ? ldy : pw.M;
        for (int64_t m = 0; m < rows; ++m) {
            const float v = raw[(size_t) (m + ldy * n)];
            if (m < pw.M) {
                res.out[(size_t) (m + pw.M * n)] = v;
            } else if (std::memcmp(&v, &DST_SENTINEL, sizeof(float)) != 0) {
                fprintf(stderr, "FAIL: %s: dst padding at row %lld of column %lld was written (%g)\n", label.c_str(),
                        (long long) m, (long long) n, (double) v);
                return res;
            }
        }
    }

    if (!ggml_sycl_test_scratchpad_site_counts(SITE, &res.calls, &res.declined, &res.engaged)) {
        fprintf(stderr, "FAIL: %s: the seam does not know the site %s\n", label.c_str(), SITE);
        return res;
    }
    ggml_sycl_test_inject_scratchpad_decline(SITE, 0);
    res.cache_entries    = get_dnnl_primitive_cache().size();
    res.scratchpad_bytes = get_dnnl_primitive_cache().get_max_scratchpad_size();
    res.ok               = true;
    return res;
}

struct errors {
    // `first` is the case's first run: the route in a taken case, the native kernel in a declined one; `second` is
    // the injected decline's run of a taken case (the native kernel).
    double first_vs_bf16 = 0.0, first_vs_f32 = 0.0, second_vs_f32 = 0.0, first_vs_second = 0.0;
    bool   finite = true;
};

// The output columns the reference is computed for: the first two, the last, and an even spread between.
std::vector<int64_t> sample_columns(int64_t N) {
    std::vector<int64_t> cols = { 0, 1, N / 2, N - 1 };
    for (int64_t j = 1; j < 8; ++j) {
        cols.push_back(j * N / 8);
    }
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    cols.erase(std::remove_if(cols.begin(), cols.end(), [N](int64_t c) { return c < 0 || c >= N; }), cols.end());
    return cols;
}

errors compare(const placed_weight &      pw,
               int64_t                    N,
               const std::vector<float> & first,
               const std::vector<float> & second) {
    errors        e;
    const int64_t K = pw.K, M = pw.M;
    for (float v : first) {
        e.finite = e.finite && std::isfinite(v);
    }
    for (float v : second) {
        e.finite = e.finite && std::isfinite(v);
    }
    std::vector<double> xf((size_t) K), xb((size_t) K);
    for (int64_t n : sample_columns(N)) {
        for (int64_t k = 0; k < K; ++k) {
            const float x  = x_value(K, k, n);
            xf[(size_t) k] = x;
            xb[(size_t) k] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
        }
        for (int64_t m = 0; m < M; ++m) {
            const float * w     = pw.dec.data() + m * K;
            double        ref_f = 0.0, ref_b = 0.0, scale = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                ref_f += w[k] * xf[(size_t) k];
                ref_b += w[k] * xb[(size_t) k];
                scale += std::fabs(w[k] * xf[(size_t) k]);
            }
            scale           = scale > 0.0 ? scale : 1.0;
            const size_t i  = (size_t) (m + M * n);
            const double r  = first[i];
            e.first_vs_bf16 = std::fmax(e.first_vs_bf16, std::fabs(r - ref_b) / scale);
            e.first_vs_f32  = std::fmax(e.first_vs_f32, std::fabs(r - ref_f) / scale);
            if (!second.empty()) {
                const double s    = second[i];
                e.second_vs_f32   = std::fmax(e.second_vs_f32, std::fabs(s - ref_f) / scale);
                e.first_vs_second = std::fmax(e.first_vs_second, std::fabs(r - s) / scale);
            }
        }
    }
    return e;
}

// A case the route does not take: the native kernel computes it and gemm() is never called.
bool check_declined(const std::string & label, const run_result & first, const errors & e) {
    bool ok = true;
    printf("  native: calls=%llu declined=%llu engaged=%llu\n", (unsigned long long) first.calls,
           (unsigned long long) first.declined, (unsigned long long) first.engaged);
    printf("  max err/S: native vs ref(f32 x) %.3e\n", e.first_vs_f32);
    if (first.calls != 0 || first.declined != 0 || first.engaged != 0) {
        fprintf(stderr, "FAIL: %s: the route reached gemm() for a node it must not take\n", label.c_str());
        ok = false;
    }
    if (!(e.first_vs_f32 <= TOL_NATIVE)) {
        fprintf(stderr, "FAIL: %s: the native kernel differs from the double reference (%.3e > %.0e)\n", label.c_str(),
                e.first_vs_f32, TOL_NATIVE);
        ok = false;
    }
    return ok;
}

bool check_taken(const std::string & label,
                 const run_result &  first,
                 const run_result &  fallen,
                 const run_result &  late,
                 const errors &      e) {
    bool ok = true;
    printf("  route: calls=%llu declined=%llu engaged=%llu | primitive cache entries=%zu scratchpad=%zu bytes\n",
           (unsigned long long) first.calls, (unsigned long long) first.declined, (unsigned long long) first.engaged,
           first.cache_entries, first.scratchpad_bytes);
    printf("  query declined: calls=%llu declined=%llu engaged=%llu\n", (unsigned long long) fallen.calls,
           (unsigned long long) fallen.declined, (unsigned long long) fallen.engaged);
    printf("  GEMM declined after the conversion: calls=%llu declined=%llu engaged=%llu\n",
           (unsigned long long) late.calls, (unsigned long long) late.declined, (unsigned long long) late.engaged);
    printf(
        "  max err/S: route vs ref(bf16 x) %.3e, route vs ref(f32 x) %.3e, native vs ref(f32 x) %.3e, route vs native "
        "%.3e\n",
        e.first_vs_bf16, e.first_vs_f32, e.second_vs_f32, e.first_vs_second);
    if (first.calls != 2 || first.engaged != 1 || first.declined != 0) {
        fprintf(stderr,
                "FAIL: %s: the route did not ask gemm() once with query_only and make one GEMM submission (want "
                "calls=2 declined=0 engaged=1)\n",
                label.c_str());
        ok = false;
    }
    if (first.cache_entries != 1) {
        fprintf(stderr,
                "FAIL: %s: the primitive cache holds %zu entries after the route run, not the route's one: the cache "
                "this test clears and reads is not the one the backend used\n",
                label.c_str(), first.cache_entries);
        ok = false;
    }
    if (fallen.calls != 1 || fallen.declined != 1 || fallen.engaged != 0) {
        fprintf(stderr,
                "FAIL: %s: after a declined query the route went on (want calls=1 declined=1 engaged=0: no conversion, "
                "no GEMM)\n",
                label.c_str());
        ok = false;
    }
    if (late.calls != 2 || late.declined != 1 || late.engaged != 0) {
        fprintf(stderr,
                "FAIL: %s: after a declined GEMM the native kernel did not compute the op alone (want calls=2 "
                "declined=1 engaged=0)\n",
                label.c_str());
        ok = false;
    }
    if (fallen.out.size() != late.out.size() ||
        std::memcmp(fallen.out.data(), late.out.data(), fallen.out.size() * sizeof(float)) != 0) {
        fprintf(stderr,
                "FAIL: %s: the native kernel's output after a declined GEMM differs from its output after a declined "
                "query: the conversion's write reached dst\n",
                label.c_str());
        ok = false;
    }
    if (!(e.first_vs_bf16 <= TOL_ROUTE)) {
        fprintf(stderr, "FAIL: %s: the route differs from the double reference over bf16(x) (%.3e > %.0e)\n",
                label.c_str(), e.first_vs_bf16, TOL_ROUTE);
        ok = false;
    }
    if (!(e.first_vs_bf16 < e.first_vs_f32)) {
        fprintf(stderr,
                "FAIL: %s: the route is no closer to the reference over bf16(x) than to the one over F32 x: the GEMM "
                "did not read the rounded copy\n",
                label.c_str());
        ok = false;
    }
    if (!(e.second_vs_f32 <= TOL_NATIVE)) {
        fprintf(stderr, "FAIL: %s: the native kernel differs from the double reference (%.3e > %.0e)\n", label.c_str(),
                e.second_vs_f32, TOL_NATIVE);
        ok = false;
    }
    if (!(e.first_vs_second <= TOL_ROUTE_VS_NATIVE)) {
        fprintf(stderr, "FAIL: %s: the route differs from the native kernel (%.3e > %.0e)\n", label.c_str(),
                e.first_vs_second, TOL_ROUTE_VS_NATIVE);
        ok = false;
    }
    return ok;
}

struct weight_spec {
    const char *         name;
    int64_t              K, M;
    std::vector<int64_t> widths;
};

}  // namespace

int main(int argc, char ** argv) {
    // See tests/sycl-selector-fallback.hpp: a plain setenv() in main() is too late for libccl's static initializer.
    sycl_test_selector_fallback(argv, "level_zero:1");
    const bool expect_opt_out = argc > 1 && std::strcmp(argv[1], "--expect-opt-out") == 0;

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr,
                "SKIP: no SYCL GPU device available -- NO DEVICE WORK WAS PERFORMED.\n"
                "      source /opt/intel/oneapi/setvars.sh --force and re-run.\n");
        return LLAMA_TEST_EXIT_SKIP;
    }
    char description[256] = {};
    ggml_backend_sycl_get_device_description(0, description, sizeof(description));
    const char * selector = getenv("ONEAPI_DEVICE_SELECTOR");
    const char * pp       = getenv("GGML_SYCL_ONEDNN_PP");
    printf("device 0: %s (ONEAPI_DEVICE_SELECTOR=%s, GGML_SYCL_ONEDNN_PP=%s)%s\n", description,
           selector ? selector : "(unset)", pp ? pp : "(unset)", expect_opt_out ? " expecting the opt-out" : "");
    printf(
        "tolerances, scaled by sum|w*x|: route against double over bf16(x) %.0e, native against double over F32 x "
        "%.0e, route against native %.0e; the route takes more than %lld columns\n",
        TOL_ROUTE, TOL_NATIVE, TOL_ROUTE_VS_NATIVE, (long long) ggml_sycl_bf16::SKINNY_MAX_COLS);
    fflush(stdout);

    bool ok = true;
    // The route's gate: the oneDNN PP admission for BF16, on by default, off with GGML_SYCL_ONEDNN_PP=0.
    if (ggml_sycl_onednn_pp_type_admitted(GGML_TYPE_BF16) == expect_opt_out) {
        fprintf(stderr, "FAIL: the oneDNN PP admission for BF16 is %s, but this run expects the route %s\n",
                expect_opt_out ? "on" : "off", expect_opt_out ? "off" : "on");
        ok = false;
    }

    const weight_spec specs[] = {
        { "blk.0.ffn_up.weight",             2304,  3072, { 1, 8, 9, 129, 512, 2048 } },
        { "blk.0.ffn_down.weight",           3072,  2304, { 9, 129, 512, 2048 }       },
        { "blk.0.hc_inject.weight",          10240, 4,    { 9, 512 }                  },
        { "blk.0.ssm_alpha.weight",          2048,  48,   { 9, 512 }                  },
        { "blk.0.hc_down.weight",            4096,  320,  { 9, 512 }                  },
        { "blk.0.ffn_gate_inp_shexp.weight", 2048,  1,    { 9, 512 }                  },
    };

    // A case the route takes: the route's run, then the same case with a decline injected on the query (1) and on the
    // GEMM after the conversion (2).
    const auto run_taken = [&](const placed_weight & pw, int64_t N, const std::string & label, case_layout layout) {
        const run_result first = run_case(backend, pw, N, 0, label, layout);
        if (!first.ok) {
            return false;
        }
        const run_result fallen = run_case(backend, pw, N, 1, label + " (query declined)", layout);
        if (!fallen.ok) {
            return false;
        }
        const run_result late = run_case(backend, pw, N, 2, label + " (GEMM declined)", layout);
        if (!late.ok) {
            return false;
        }
        const errors e       = compare(pw, N, first.out, fallen.out);
        const bool   case_ok = check_taken(label, first, fallen, late, e) && e.finite;
        if (!e.finite) {
            fprintf(stderr, "FAIL: %s: a non-finite output\n", label.c_str());
        }
        return case_ok;
    };
    // A case on an empty planned buffer the route must decline: no gemm() call, one under-estimate counted against the
    // buffer's cohort, and the native kernel's output.
    const auto run_short = [&](const placed_weight & pw, int64_t N, const std::string & label) {
        const size_t     before = ggml_sycl::zone_sizing_underestimate_count(SRC1_COHORT);
        const run_result first  = run_case(backend, pw, N, 0, label);
        if (!first.ok) {
            return false;
        }
        const size_t counted = ggml_sycl::zone_sizing_underestimate_count(SRC1_COHORT) - before;
        printf("  under-estimates counted against %s: %zu\n", SRC1_COHORT, counted);
        const errors e       = compare(pw, N, first.out, {});
        bool         case_ok = check_declined(label, first, e);
        if (counted != 1) {
            fprintf(stderr, "FAIL: %s: the route's decline on a short buffer counted %zu under-estimates, not 1\n",
                    label.c_str(), counted);
            case_ok = false;
        }
        if (!e.finite) {
            fprintf(stderr, "FAIL: %s: a non-finite output\n", label.c_str());
            case_ok = false;
        }
        return case_ok;
    };
    const auto report = [&](bool case_ok) {
        printf("  %s\n", case_ok ? "ok" : "FAILED");
        fflush(stdout);
        ok = ok && case_ok;
    };

    int64_t salt = 11;
    for (const weight_spec & spec : specs) {
        const std::unique_ptr<placed_weight> pw = place_weight(spec.name, spec.K, spec.M, salt++);
        if (!pw) {
            fprintf(stderr, "FAIL: could not place the weight %s on the SYCL device\n", spec.name);
            ok = false;
            continue;
        }
        // The first two cases of the run, while the planned buffer is still empty.
        if (&spec == &specs[0] && !expect_opt_out) {
            std::string label = std::string(spec.name) + " N=129, walk leaves the route out";
            printf("case %-48s K=%lld M=%lld: running\n", label.c_str(), (long long) spec.K, (long long) spec.M);
            fflush(stdout);
            ggml_sycl::test_set_bf16_onednn_walk_counted(false);
            bool case_ok = run_short(*pw, 129, label);
            ggml_sycl::test_set_bf16_onednn_walk_counted(true);
            report(case_ok);

            size_t act_bytes = 0;
            if (!ggml_sycl::zone_dequant_f16_region_bytes(spec.K * 129, &act_bytes) || act_bytes == 0) {
                fprintf(stderr, "FAIL: the route's copy size for K=%lld N=129 does not compute\n", (long long) spec.K);
                ok = false;
                continue;
            }
            label = std::string(spec.name) + " N=129, zone below the route's copy";
            printf("case %-48s K=%lld M=%lld: running\n", label.c_str(), (long long) spec.K, (long long) spec.M);
            fflush(stdout);
            ggml_sycl::test_set_dequant_f16_walk_limit(act_bytes - 1);
            case_ok = run_short(*pw, 129, label);
            ggml_sycl::test_set_dequant_f16_walk_limit(SIZE_MAX);
            report(case_ok);
        }
        for (const int64_t N : spec.widths) {
            const bool        taken = N > ggml_sycl_bf16::SKINNY_MAX_COLS && !expect_opt_out;
            const std::string label = std::string(spec.name) + " N=" + std::to_string(N);
            printf("case %-48s K=%lld M=%lld %s: running\n", label.c_str(), (long long) spec.K, (long long) spec.M,
                   taken ? "(route)" : "(native)");
            fflush(stdout);
            if (taken) {
                report(run_taken(*pw, N, label, CASE_LAYOUT_PLAIN));
                continue;
            }
            const run_result first = run_case(backend, *pw, N, 0, label);
            if (!first.ok) {
                ok = false;
                continue;
            }
            const errors e       = compare(*pw, N, first.out, {});
            bool         case_ok = check_declined(label, first, e);
            if (!e.finite) {
                fprintf(stderr, "FAIL: %s: a non-finite output\n", label.c_str());
                case_ok = false;
            }
            report(case_ok);
        }
        // The layouts the route reads through strides, on the up projection.
        if (&spec == &specs[0] && !expect_opt_out) {
            const struct {
                case_layout  layout;
                const char * name;
            } layouts[] = {
                { CASE_LAYOUT_STRIDED_SRC1, "src1 columns wider than K" },
                { CASE_LAYOUT_STRIDED_DST,  "dst columns wider than M"  },
                { CASE_LAYOUT_SRC1_3D,      "src1 of three planes"      },
            };

            for (const auto & l : layouts) {
                const std::string label = std::string(spec.name) + " N=129, " + l.name;
                printf("case %-48s K=%lld M=%lld (route): running\n", label.c_str(), (long long) spec.K,
                       (long long) spec.M);
                fflush(stdout);
                report(run_taken(*pw, 129, label, l.layout));
            }
        }
    }

    ggml_backend_free(backend);
    if (!ok) {
        fprintf(stderr, "test-sycl-bf16-onednn-mul-mat-device: FAIL\n");
        return 1;
    }
    printf("test-sycl-bf16-onednn-mul-mat-device: PASS\n");
    return 0;
}
