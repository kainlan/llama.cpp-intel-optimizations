// ggml_vec_dot_q2_0_q8_0 against an independent oracle (llama.cpp-tjg8).
//
// Q2_0: block_q2_0 = fp16 d + 16 B of 2-bit codes per 64 weights, byte b holding
// weights 4b..4b+3 low bits first, code c -> (c - 1) * d, i.e. {-1, 0, +1, +2}.
// vec_dot_type is Q8_0, so one Q2_0 block pairs with two consecutive Q8_0 blocks.
//
// The oracle is written from that format description in double precision and does
// not call ggml_vec_dot_q2_0_q8_0_generic, so a kernel and its scalar reference
// cannot agree by being the same code. On x86 with AVX2 the test also requires
// that the registered vec_dot is NOT the scalar reference: before llama.cpp-tjg8
// arch-fallback.h renamed the scalar version to ggml_vec_dot_q2_0_q8_0 on x86, so
// the accuracy checks alone pass trivially there and prove nothing about a SIMD
// path.

#include "../ggml/src/ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Weak so that a build where the scalar version is renamed away (the x86 state
// before the SIMD kernel exists) reports a runtime FAIL instead of a link error.
extern "C" __attribute__((weak)) void ggml_vec_dot_q2_0_q8_0_generic(
        int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static int g_failed = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__);   \
            std::fprintf(stderr, "\n");                   \
            g_failed++;                                   \
        }                                                 \
    } while (0)

// d * (code - 1) * d1 * q, accumulated in double; `mag` is the sum of absolute
// terms, the scale a float accumulation error is relative to.
static double oracle_dot(int n, const block_q2_0 * x, const block_q8_0 * y, double * mag) {
    double sum = 0.0;
    double m   = 0.0;
    for (int i = 0; i < n / 64; i++) {
        const double d0 = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < 64; j++) {
            const int    code = (x[i].qs[j / 4] >> (2 * (j % 4))) & 3;
            const double d1   = ggml_fp16_to_fp32(y[2 * i + j / 32].d);
            const double term = d0 * (code - 1) * d1 * y[2 * i + j / 32].qs[j % 32];
            sum += term;
            m += std::fabs(term);
        }
    }
    *mag = m;
    return sum;
}

enum fill_mode {
    FILL_RANDOM,    // random codes, random y
    FILL_MAX_POS,   // every code 3 (+2), y = 127
    FILL_MAX_NEG,   // every code 3 (+2), y = -128
    FILL_ALL_ZERO,  // every code 0 (-1), y = -128
    FILL_ZERO_CODE, // every code 1 (0), random y
};

static void fill(fill_mode mode, std::mt19937 & rng, int nb, block_q2_0 * x, block_q8_0 * y, float d0, float d1) {
    for (int i = 0; i < nb; i++) {
        x[i].d = ggml_fp32_to_fp16(d0);
        for (int j = 0; j < 16; j++) {
            uint8_t b;
            switch (mode) {
                case FILL_MAX_POS:
                case FILL_MAX_NEG:   b = 0xFF; break;
                case FILL_ALL_ZERO:  b = 0x00; break;
                case FILL_ZERO_CODE: b = 0x55; break;
                default:             b = (uint8_t) rng(); break;
            }
            x[i].qs[j] = b;
        }
    }
    for (int i = 0; i < 2 * nb; i++) {
        y[i].d = ggml_fp32_to_fp16(d1);
        for (int j = 0; j < 32; j++) {
            switch (mode) {
                case FILL_MAX_POS:  y[i].qs[j] = 127; break;
                case FILL_MAX_NEG:
                case FILL_ALL_ZERO: y[i].qs[j] = -128; break;
                default:            y[i].qs[j] = (int8_t) (rng() & 0xFF); break;
            }
        }
    }
}

static void run_case(const char * name, fill_mode mode, int nb, float d0, float d1, size_t misalign, std::mt19937 & rng) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);

    // Misaligned views: rows of a tensor sit at multiples of 18 B, so the kernel
    // must not assume any alignment of either operand.
    std::vector<uint8_t> xb(sizeof(block_q2_0) * nb + 64);
    std::vector<uint8_t> yb(sizeof(block_q8_0) * 2 * nb + 64);
    block_q2_0 * x = (block_q2_0 *) (xb.data() + misalign);
    block_q8_0 * y = (block_q8_0 *) (yb.data() + misalign);
    fill(mode, rng, nb, x, y, d0, d1);

    // Bytes past the operands are poisoned so a read past the end that leaks into
    // the result shows up as a wrong sum rather than a lucky zero.
    std::memset(xb.data() + misalign + sizeof(block_q2_0) * nb, 0x7F, 64 - misalign);
    std::memset(yb.data() + misalign + sizeof(block_q8_0) * 2 * nb, 0x7F, 64 - misalign);

    double       mag = 0.0;
    const double ref = oracle_dot(nb * 64, x, y, &mag);

    float got = -12345.0f;
    tr->vec_dot(nb * 64, &got, 0, x, 0, y, 0, 1);

    const double tol = 2e-5 * mag + 1e-30;
    const double err = std::fabs((double) got - ref);
    CHECK(std::isfinite(got) && err <= tol, "%s nb=%d mis=%zu: got %.9g oracle %.9g err %.3g tol %.3g", name, nb,
          misalign, (double) got, ref, err, tol);
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(42);

#if defined(__x86_64__) || defined(_M_X64)
    if (ggml_cpu_has_avx2()) {
        const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
        CHECK(&ggml_vec_dot_q2_0_q8_0_generic != nullptr &&
                      (void *) tr->vec_dot != (void *) ggml_vec_dot_q2_0_q8_0_generic,
              "x86 AVX2 host: Q2_0 vec_dot is the scalar reference, there is no SIMD kernel to test");
    }
#endif

    // 1 and the odd counts exercise any 2-block unrolling and its tail; 64 is the
    // Qwen3.8 expert row length (4096 / 64).
    static const int    nbs[] = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 127, 256 };
    static const size_t mis[] = { 0, 1, 3 };

    for (int nb : nbs) {
        for (size_t m : mis) {
            run_case("random",         FILL_RANDOM,    nb, 0.37f,    0.011f,   m, rng);
            run_case("random-big-d",   FILL_RANDOM,    nb, 65504.0f, 65504.0f, m, rng);
            run_case("random-tiny-d",  FILL_RANDOM,    nb, 6.0e-8f,  6.0e-8f,  m, rng);
            run_case("random-zero-d0", FILL_RANDOM,    nb, 0.0f,     0.5f,     m, rng);
            run_case("random-zero-d1", FILL_RANDOM,    nb, 0.5f,     0.0f,     m, rng);
            run_case("max-pos",        FILL_MAX_POS,   nb, 1.0f,     1.0f,     m, rng);
            run_case("max-neg",        FILL_MAX_NEG,   nb, 1.0f,     1.0f,     m, rng);
            run_case("all-code0",      FILL_ALL_ZERO,  nb, 1.0f,     1.0f,     m, rng);
            run_case("all-code1",      FILL_ZERO_CODE, nb, 1.0f,     1.0f,     m, rng);
        }
    }

    // Many random rows at the row length the experts use, so a rare data-dependent
    // error (a lane, a code, a sign) cannot hide behind one lucky row.
    for (int r = 0; r < 200; r++) {
        run_case("random-row", FILL_RANDOM, 64, 0.01f + 0.001f * r, 0.002f + 0.0003f * r, (size_t) (r % 3), rng);
    }

    if (g_failed) {
        std::fprintf(stderr, "%d check(s) FAILED\n", g_failed);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
