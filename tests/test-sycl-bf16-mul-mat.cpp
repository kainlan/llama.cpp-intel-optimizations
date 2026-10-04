// llama.cpp-9qjy: the native BF16-weight x F32-activation MUL_MAT kernel and the
// shape contract that admits it.
//
// Compiles ggml-sycl/mul-mat-bf16.hpp directly and runs it on whatever device
// ONEAPI_DEVICE_SELECTOR picks: the registration pins opencl:cpu, so this needs
// no GPU. Run it with the selector set to a GPU to check the same kernels there.
//
// Reference: double-precision sum over the BF16-decoded weights and the F32
// activations. The kernel accumulates in F32, so the tolerance scales with the
// sum of |w * x|, which bounds the F32 rounding error of any summation order.

#include "ggml-sycl/mul-mat-bf16.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++g_failures;                                    \
        }                                                    \
    } while (0)

struct run_case {
    const char * name;
    int64_t      K;
    int64_t      M;
    int64_t      N;
    int64_t      ldx_pad;   // extra elements per activation column
    int64_t      ldy_pad;   // extra elements per output column
    int64_t      w_offset;  // elements skipped at the weight base (misaligns the 8-byte loads)
    float        w_scale;   // weight magnitude
};

static void run_one(sycl::queue & q, const run_case & c) {
    const int64_t ldx = c.K + c.ldx_pad;
    const int64_t ldy = c.M + c.ldy_pad;

    std::mt19937                          rng(1234 + static_cast<unsigned>(c.K * 31 + c.M * 7 + c.N));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Weights as BF16 bits, decoded once for the reference.
    const size_t          w_elems = static_cast<size_t>(c.M * c.K + c.w_offset);
    std::vector<uint16_t> w_bits(w_elems);
    std::vector<float>    w_dec(w_elems);
    for (size_t i = 0; i < w_elems; ++i) {
        const ggml_bf16_t b = ggml_fp32_to_bf16(dist(rng) * c.w_scale);
        w_bits[i]           = b.bits;
        w_dec[i]            = ggml_bf16_to_fp32(b);
    }
    std::vector<float> x(static_cast<size_t>(ldx * c.N));
    for (float & v : x) {
        v = dist(rng);
    }

    uint16_t * w_dev = sycl::malloc_device<uint16_t>(w_elems, q);
    float *    x_dev = sycl::malloc_device<float>(x.size(), q);
    float *    y_dev = sycl::malloc_device<float>(static_cast<size_t>(ldy * c.N), q);
    q.memcpy(w_dev, w_bits.data(), w_elems * sizeof(uint16_t));
    q.memcpy(x_dev, x.data(), x.size() * sizeof(float));
    // Sentinel: padding elements of y must come back untouched.
    std::vector<float> y_init(static_cast<size_t>(ldy * c.N), -12345.0f);
    q.memcpy(y_dev, y_init.data(), y_init.size() * sizeof(float));
    q.wait();

    ggml_sycl_bf16::mul_mat_bf16_f32(q, w_dev + c.w_offset, x_dev, y_dev, c.K, c.M, c.N, ldx, ldy).wait();

    std::vector<float> y(y_init.size());
    q.memcpy(y.data(), y_dev, y.size() * sizeof(float)).wait();

    int    bad     = 0;
    double max_err = 0.0;
    for (int64_t n = 0; n < c.N; ++n) {
        for (int64_t m = 0; m < c.M; ++m) {
            double ref = 0.0;
            double mag = 0.0;
            for (int64_t k = 0; k < c.K; ++k) {
                const double p = static_cast<double>(w_dec[c.w_offset + m * c.K + k]) * x[n * ldx + k];
                ref += p;
                mag += std::fabs(p);
            }
            const double got = y[n * ldy + m];
            const double err = std::fabs(got - ref);
            const double tol = 1e-5 * mag + 1e-30;
            max_err          = std::fmax(max_err, mag > 0.0 ? err / mag : err);
            if (!(err <= tol)) {
                if (bad < 3) {
                    std::printf("  %s: m=%lld n=%lld got=%.9g ref=%.9g err=%.3g tol=%.3g\n", c.name, (long long) m,
                                (long long) n, got, ref, err, tol);
                }
                ++bad;
            }
        }
        // Padding rows between columns of y must be untouched.
        for (int64_t m = c.M; m < ldy; ++m) {
            if (y[n * ldy + m] != -12345.0f) {
                if (bad < 3) {
                    std::printf("  %s: padding y[%lld,%lld] was overwritten\n", c.name, (long long) m, (long long) n);
                }
                ++bad;
            }
        }
    }
    CHECK(bad == 0, "%s: %d bad outputs (K=%lld M=%lld N=%lld)", c.name, bad, (long long) c.K, (long long) c.M,
          (long long) c.N);
    std::printf("  %-34s K=%-5lld M=%-4lld N=%-4lld max_rel_err=%.3g %s\n", c.name, (long long) c.K, (long long) c.M,
                (long long) c.N, max_err, bad == 0 ? "ok" : "BAD");

    sycl::free(w_dev, q);
    sycl::free(x_dev, q);
    sycl::free(y_dev, q);
}

// ---- shape contract -------------------------------------------------------

static ggml_tensor * make(ggml_context * ctx, ggml_type t, int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
    return ggml_new_tensor_4d(ctx, t, a, b, c, d);
}

static void check_shape_contract() {
    ggml_init_params p{};
    p.mem_size         = 4 * 1024 * 1024;
    p.mem_buffer       = nullptr;
    p.no_alloc         = true;
    ggml_context * ctx = ggml_init(p);
    CHECK(ctx != nullptr, "ggml_init");

    ggml_tensor * w = make(ctx, GGML_TYPE_BF16, 64, 32);
    ggml_tensor * x = make(ctx, GGML_TYPE_F32, 64, 5);
    ggml_tensor * y = make(ctx, GGML_TYPE_F32, 32, 5);
    using ggml_sycl_bf16::mul_mat_shape_supported;
    CHECK(mul_mat_shape_supported(w, x, y), "plain 2-D weight x contiguous activations is the supported shape");

    // F16 / F32 weights are not this executor's.
    CHECK(!mul_mat_shape_supported(make(ctx, GGML_TYPE_F16, 64, 32), x, y), "F16 weight must be refused");
    // Non-F32 activations / dst.
    CHECK(!mul_mat_shape_supported(w, make(ctx, GGML_TYPE_F16, 64, 5), y), "F16 activations must be refused");
    CHECK(!mul_mat_shape_supported(w, x, make(ctx, GGML_TYPE_F16, 32, 5)), "F16 dst must be refused");
    // Batched weight (broadcast src0) is not handled.
    CHECK(!mul_mat_shape_supported(make(ctx, GGML_TYPE_BF16, 64, 32, 2), make(ctx, GGML_TYPE_F32, 64, 5, 2),
                                   make(ctx, GGML_TYPE_F32, 32, 5, 2)),
          "batched weight must be refused");
    // K mismatch / M mismatch.
    CHECK(!mul_mat_shape_supported(w, make(ctx, GGML_TYPE_F32, 63, 5), y), "K mismatch must be refused");
    CHECK(!mul_mat_shape_supported(w, x, make(ctx, GGML_TYPE_F32, 31, 5)), "M mismatch must be refused");
    // Permuted / non-contiguous weight.
    {
        ggml_tensor * wt = make(ctx, GGML_TYPE_BF16, 64, 32);
        wt->nb[1]        = wt->nb[0] * 64 + wt->nb[0];  // padded rows
        CHECK(!mul_mat_shape_supported(wt, x, y), "row-padded weight must be refused");
    }
    // Activations with a padded column stride are fine (uniform stride) ...
    {
        ggml_tensor * xp = make(ctx, GGML_TYPE_F32, 64, 5);
        xp->nb[1]        = sizeof(float) * 72;
        xp->nb[2]        = xp->nb[1] * 5;
        xp->nb[3]        = xp->nb[2];
        CHECK(mul_mat_shape_supported(w, xp, y), "uniformly padded activation columns must be accepted");
    }
    // ... a 3-D activation tensor whose planes do not continue the column stride is not.
    {
        ggml_tensor * x3 = make(ctx, GGML_TYPE_F32, 64, 5, 3);
        ggml_tensor * y3 = make(ctx, GGML_TYPE_F32, 32, 5, 3);
        CHECK(mul_mat_shape_supported(w, x3, y3), "contiguous 3-D activations flatten to one column run");
        x3->nb[2] = x3->nb[1] * 5 + 64;  // gap between planes
        x3->nb[3] = x3->nb[2] * 3;
        CHECK(!mul_mat_shape_supported(w, x3, y3), "a gap between activation planes must be refused");
    }
    // A single column has no stride to check, whatever nb[1] says.
    {
        ggml_tensor * x1 = make(ctx, GGML_TYPE_F32, 64, 1);
        ggml_tensor * y1 = make(ctx, GGML_TYPE_F32, 32, 1);
        x1->nb[1]        = 12345;
        CHECK(mul_mat_shape_supported(w, x1, y1), "a single activation column is always accepted");
    }
    // Strided elements within a row are not indexable as t[col * ld + row].
    {
        ggml_tensor * xs = make(ctx, GGML_TYPE_F32, 64, 5);
        xs->nb[0]        = 8;
        CHECK(!mul_mat_shape_supported(w, xs, y), "strided activation elements must be refused");
    }
    CHECK(!mul_mat_shape_supported(nullptr, x, y) && !mul_mat_shape_supported(w, nullptr, y) &&
              !mul_mat_shape_supported(w, x, nullptr),
          "null operands must be refused");

    ggml_free(ctx);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    check_shape_contract();

    sycl::queue q{ sycl::default_selector_v };
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const run_case cases[] = {
        // name                        K     M    N   ldx_pad ldy_pad w_off scale
        { "decode router",          2048, 256, 1,   0, 0, 0, 1.0f   },
        { "decode padded strides",  2048, 40,  1,   3, 5, 0, 1.0f   },
        { "skinny n=2",             2048, 33,  2,   0, 0, 0, 1.0f   },
        { "skinny n=5",             1024, 17,  5,   4, 3, 0, 1.0f   },
        { "skinny n=8",             2048, 100, 8,   0, 0, 0, 1.0f   },
        { "K not multiple of 4",    2050, 7,   1,   0, 0, 0, 1.0f   },
        { "K=3 (below sub-group)",  3,    9,   1,   0, 0, 0, 1.0f   },
        { "misaligned weight base", 2048, 12,  3,   0, 0, 1, 1.0f   },
        { "M=1",                    4096, 1,   1,   0, 0, 0, 1.0f   },
        { "tiled n=9",              2048, 64,  9,   0, 0, 0, 1.0f   },
        { "tiled n=64 ragged m",    2048, 77,  64,  0, 0, 0, 1.0f   },
        { "tiled n=100 ragged all", 130,  45,  100, 7, 9, 0, 1.0f   },
        { "tiled K<tile",           20,   33,  33,  0, 0, 0, 1.0f   },
        { "tiled n=512 large",      2048, 512, 512, 0, 0, 0, 1.0f   },
        // Magnitudes far outside F16 range (max 65504): BF16 -> F32 must be exact, never an F16 detour.
        { "huge weights (1e20)",    256,  16,  1,   0, 0, 0, 1e20f  },
        { "huge weights tiled",     256,  40,  16,  0, 0, 0, 1e20f  },
        { "tiny weights (1e-30)",   256,  16,  1,   0, 0, 0, 1e-30f },
    };
    for (const run_case & c : cases) {
        run_one(q, c);
    }

    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
