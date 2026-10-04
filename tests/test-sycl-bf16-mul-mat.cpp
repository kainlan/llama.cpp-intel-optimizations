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
#include <cstring>
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
    int64_t      x_offset;  // elements skipped at the activation base (misaligns the float4 loads)
};

// Guard regions around every buffer the kernel touches. They are poisoned with values that
// corrupt the result if read (NaN) or are detectably changed if written, so an out-of-bounds
// access by any lane of any split fails the case instead of passing by luck.
constexpr int64_t GUARD = 64;

static void run_one(sycl::queue & q, const run_case & c) {
    const int64_t ldx = c.K + c.ldx_pad;
    const int64_t ldy = c.M + c.ldy_pad;

    std::mt19937                          rng(1234 + static_cast<unsigned>(c.K * 31 + c.M * 7 + c.N));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Host images, each laid out as [guard][offset][data][guard]. Guards and the stride padding
    // between activation columns are NaN, so a read of anything outside the K x N / M x K
    // extents poisons the output. The weight base offset sits inside the allocation (its
    // elements are real data the kernel must simply not index).
    const uint16_t        w_nan   = 0x7FC0;
    const size_t          w_elems = static_cast<size_t>(c.M * c.K + c.w_offset);
    std::vector<uint16_t> w_bits(w_elems + 2 * GUARD, w_nan);
    std::vector<float>    w_dec(w_elems);
    for (size_t i = 0; i < w_elems; ++i) {
        const ggml_bf16_t b = ggml_fp32_to_bf16(dist(rng) * c.w_scale);
        w_bits[GUARD + i]   = b.bits;
        w_dec[i]            = ggml_bf16_to_fp32(b);
    }
    const size_t       x_elems = static_cast<size_t>(c.x_offset + ldx * c.N);
    std::vector<float> x_host(x_elems + 2 * GUARD, std::nanf(""));
    float *            x = x_host.data() + GUARD + c.x_offset;  // column 0, element 0
    for (int64_t n = 0; n < c.N; ++n) {
        for (int64_t k = 0; k < c.K; ++k) {
            x[n * ldx + k] = dist(rng);
        }
    }

    const size_t       y_elems  = static_cast<size_t>(ldy * c.N);
    const float        y_poison = -12345.0f;
    std::vector<float> y_init(y_elems + 2 * GUARD, y_poison);

    uint16_t * w_dev = sycl::malloc_device<uint16_t>(w_bits.size(), q);
    float *    x_dev = sycl::malloc_device<float>(x_host.size(), q);
    float *    y_dev = sycl::malloc_device<float>(y_init.size(), q);
    q.memcpy(w_dev, w_bits.data(), w_bits.size() * sizeof(uint16_t));
    q.memcpy(x_dev, x_host.data(), x_host.size() * sizeof(float));
    // Sentinel: padding elements of y must come back untouched.
    q.memcpy(y_dev, y_init.data(), y_init.size() * sizeof(float));
    q.wait();

    ggml_sycl_bf16::mul_mat_bf16_f32(q, w_dev + GUARD + c.w_offset, x_dev + GUARD + c.x_offset, y_dev + GUARD, c.K, c.M,
                                     c.N, ldx, ldy)
        .wait();

    std::vector<float> y_all(y_init.size());
    q.memcpy(y_all.data(), y_dev, y_all.size() * sizeof(float)).wait();
    std::vector<uint16_t> w_back(w_bits.size());
    std::vector<float>    x_back(x_host.size());
    q.memcpy(w_back.data(), w_dev, w_back.size() * sizeof(uint16_t)).wait();
    q.memcpy(x_back.data(), x_dev, x_back.size() * sizeof(float)).wait();
    const float * y = y_all.data() + GUARD;

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
            if (y[n * ldy + m] != y_poison) {
                if (bad < 3) {
                    std::printf("  %s: padding y[%lld,%lld] was overwritten\n", c.name, (long long) m, (long long) n);
                }
                ++bad;
            }
        }
    }
    // Guard regions around y, and the read-only inputs, must come back bit-identical.
    for (int64_t i = 0; i < GUARD; ++i) {
        if (y_all[i] != y_poison || y_all[GUARD + y_elems + i] != y_poison) {
            std::printf("  %s: y guard element %lld was written\n", c.name, (long long) i);
            ++bad;
            break;
        }
    }
    for (size_t i = 0; i < w_bits.size(); ++i) {
        if (w_back[i] != w_bits[i]) {
            std::printf("  %s: weight element %zu was written\n", c.name, i);
            ++bad;
            break;
        }
    }
    for (size_t i = 0; i < x_host.size(); ++i) {
        if (std::memcmp(&x_back[i], &x_host[i], sizeof(float)) != 0) {
            std::printf("  %s: activation element %zu was written\n", c.name, i);
            ++bad;
            break;
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

static void check_split_policy() {
    using ggml_sycl_bf16::skinny_split;
    // Short matrices split K to fill the device; tall ones keep one sub-group per row.
    CHECK(skinny_split(10240, 1) == 32, "M=1 K=10240 should use the maximum split, got %d", skinny_split(10240, 1));
    CHECK(skinny_split(10240, 4) == 32, "M=4 K=10240 should use the maximum split");
    CHECK(skinny_split(2048, 48) == 16, "M=48 K=2048 is limited by K (16 passes of 128), got %d",
          skinny_split(2048, 48));
    CHECK(skinny_split(4096, 320) == 8, "M=320 K=4096 should reach ~2048 sub-groups with a split of 8, got %d",
          skinny_split(4096, 320));
    CHECK(skinny_split(4096, 4096) == 1, "tall matrix must not split");
    // A split never leaves a sub-group with less than one 128-element pass of K.
    CHECK(skinny_split(36, 1) == 1, "K=36 must not split");
    CHECK(skinny_split(128, 1) == 1, "K=128 must not split");
    CHECK(skinny_split(256, 1) == 2, "K=256 splits in two");
    for (int64_t K : { 1, 36, 2052, 10240, 65536 }) {
        for (int64_t M : { 1, 3, 48, 320, 100000 }) {
            const int s = skinny_split(K, M);
            CHECK(s >= 1 && s <= 32 && (s & (s - 1)) == 0, "split %d for K=%lld M=%lld is not a power of two in [1,32]",
                  s, (long long) K, (long long) M);
        }
    }
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    check_shape_contract();
    check_split_policy();

    sycl::queue q{ sycl::default_selector_v };
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const run_case cases[] = {
        // name                        K     M    N   ldx_pad ldy_pad w_off scale x_off
        { "decode router",              2048,  256, 1,   0, 0, 0, 1.0f,   0 },
        { "decode padded strides",      2048,  40,  1,   3, 5, 0, 1.0f,   0 },
        { "skinny n=2",                 2048,  33,  2,   0, 0, 0, 1.0f,   0 },
        { "skinny n=5",                 1024,  17,  5,   4, 3, 0, 1.0f,   0 },
        { "skinny n=8",                 2048,  100, 8,   0, 0, 0, 1.0f,   0 },
        { "K not multiple of 4",        2050,  7,   1,   0, 0, 0, 1.0f,   0 },
        { "K=3 (below sub-group)",      3,     9,   1,   0, 0, 0, 1.0f,   0 },
        { "misaligned weight base",     2048,  12,  3,   0, 0, 1, 1.0f,   0 },
        { "M=1",                        4096,  1,   1,   0, 0, 0, 1.0f,   0 },
        // Decode shapes from Qwen3.8 (the ops that run 484 times per token): small M, large K, so
        // the skinny kernel splits K across sub-groups and reduces through local memory.
        { "qwen hc_inject M=4 K=10240", 10240, 4,   1,   0, 0, 0, 1.0f,   0 },
        { "qwen hc_inject N=2",         10240, 4,   2,   0, 0, 0, 1.0f,   0 },
        { "qwen ssm_alpha M=48",        2048,  48,  1,   0, 0, 0, 1.0f,   0 },
        { "qwen hc_down M=320",         4096,  320, 1,   0, 0, 0, 1.0f,   0 },
        { "qwen shexp gate M=1",        2048,  1,   1,   0, 0, 0, 1.0f,   0 },
        // vec4 with K / 4 not a multiple of the 16 lanes (or of a split's 16-lane pass).
        { "vec4 K=36",                  36,    5,   1,   0, 0, 0, 1.0f,   0 },
        { "vec4 K=2052",                2052,  3,   2,   0, 0, 0, 1.0f,   0 },
        { "vec4 K=2052 M=1 N=8",        2052,  1,   8,   0, 0, 0, 1.0f,   0 },
        { "vec4 K=130 (not mult of 4)", 130,   6,   1,   0, 0, 0, 1.0f,   0 },
        // Activation base / column stride that defeat float4 loads while the weights stay vec4.
        { "x base misaligned (16B)",    2048,  8,   2,   0, 0, 0, 1.0f,   1 },
        { "x stride odd, vec4 weights", 2048,  8,   3,   1, 0, 0, 1.0f,   0 },
        { "x base +2, M=4 K=10240",     10240, 4,   1,   0, 0, 0, 1.0f,   2 },
        { "tiled n=9",                  2048,  64,  9,   0, 0, 0, 1.0f,   0 },
        { "tiled n=64 ragged m",        2048,  77,  64,  0, 0, 0, 1.0f,   0 },
        { "tiled n=100 ragged all",     130,   45,  100, 7, 9, 0, 1.0f,   0 },
        { "tiled K<tile",               20,    33,  33,  0, 0, 0, 1.0f,   0 },
        { "tiled n=512 large",          2048,  512, 512, 0, 0, 0, 1.0f,   0 },
        // Magnitudes far outside F16 range (max 65504): BF16 -> F32 must be exact, never an F16 detour.
        { "huge weights (1e20)",        256,   16,  1,   0, 0, 0, 1e20f,  0 },
        { "huge weights tiled",         256,   40,  16,  0, 0, 0, 1e20f,  0 },
        { "tiny weights (1e-30)",       256,   16,  1,   0, 0, 0, 1e-30f, 0 },
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
