// The NVFP4 dot product's decode on the host: the bits of each helper it may use, and the functor
// against an independent oracle.
//
// vec_dot_nvfp4_q8_1 (vecdotq.hpp) serves every NVFP4 MMVQ launch: the dense mat-vec and the expert
// (_id) launches. Its E2M1 codes can be decoded through the kvalues_mxfp4 table
// (get_int_from_table_16, built on emulated byte permutes) or arithmetically (get_int_from_mxfp4,
// which the MXFP4 functor uses). Swapping one for the other is only correct while both return the
// same ints for every input the functor can pass, so this test compares them:
//   - every value of each byte, with the other three bytes at 0x00, at 0xff and at one mixed pattern;
//   - every combination of four low-nibble codes, under each of the 16 high nibbles.
// It also runs the functor itself for each iqs the MMVQ kernels pass it (0 and 4: QI_NVFP4 = 8,
// VDR_NVFP4_Q8_1_MMVQ = 4) and compares the result with a double-precision oracle written from the
// format (64 weights, four 16-weight sub-blocks, a UE4M3 scale each, codes from the E2M1 table).
// The oracle calls none of the code under test.
//
// The UE4M3 scale helper, ggml_sycl_ue4m3_to_fp32 (common.hpp), is compared bit for bit (memcmp, so
// the sign of zero and NaN-ness count) for all 256 bytes with two references: the value written
// from the format, and the body it had before it built the float from its bits (compares and
// sycl::ldexp), kept here and not in the header.
//
// These comparisons pass before and after a decode is swapped: they guard that the bits stay the
// same. tests/test-sycl-nvfp4-decode-source.py pins which form the code takes.
//
// The functions are plain inline code, so main() calls them directly: no queue is opened and no
// device is touched. A host comparison speaks for the device because no step in either decode
// rounds. The codes are integer arithmetic (shifts, masks, byte-wise adds). Every halved scale is
// man * 2^-10 (exp == 0) or (1 + man/8) * 2^(exp - 8) with exp 1..15, so its values run from 2^-10
// (0x01) to 1.75 * 2^7 = 224 (0x7e): man/8 and each power-of-two scaling are exact in binary32
// (ldexp is exact by definition, 0 ulp in the OpenCL spec), the bit form puts the same exponent and
// mantissa fields down directly, and the smallest nonzero value, 2^-10, is far above binary32's
// normal range floor, so a device that flushes denormals sees none.

#include "dequantize.hpp"
#include "ggml-common.h"
#include "ggml.h"
#include "vecdotq.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

static int g_failed = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
            std::fprintf(stderr, "\n");                 \
            g_failed++;                                 \
        }                                               \
    } while (0)

// E2M1 code -> value, from the format: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6, bit 3 the sign.
static const double k_e2m1[16] = { 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6 };

// UE4M3 scale from the format (bias 7): exp > 0 is (1 + man/8) * 2^(exp - 7), exp == 0 is (man/8) *
// 2^-6. 0x7f is the NaN encoding and ggml stores no NaN scale, so it reads as 0; bit 7 is not part
// of the value, so 0xff reads as 0x7f does.
static double oracle_ue4m3(uint8_t x) {
    if ((x & 0x7f) == 0x7f) {
        return 0.0;
    }
    const int exp = (x >> 3) & 0xf;
    const int man = x & 0x7;
    return exp == 0 ? std::ldexp((double) man, -9) : std::ldexp(1.0 + man / 8.0, exp - 7);
}

// The helper's body before it built the float from its bits.
static float legacy_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7f || x == 0xff) {
        return 0.0f;
    }
    const int   exp = (x >> 3) & 0xf;
    const int   man = x & 0x7;
    const float raw = exp == 0 ? sycl::ldexp((float) man, -9) : sycl::ldexp(1.0f + (float) man / 8.0f, exp - 7);
    return raw * 0.5f;
}

static bool same_bits(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// ggml stores the scale halved, because kvalues_mxfp4 holds the E2M1 values doubled.
static void check_scale_all() {
    for (int i = 0; i < 256; i++) {
        const uint8_t x      = (uint8_t) i;
        const float   got    = ggml_sycl_ue4m3_to_fp32(x);
        const float   format = (float) (0.5 * oracle_ue4m3(x));
        const float   legacy = legacy_ue4m3_to_fp32(x);
        uint32_t      bits[3];
        std::memcpy(&bits[0], &got, sizeof(float));
        std::memcpy(&bits[1], &format, sizeof(float));
        std::memcpy(&bits[2], &legacy, sizeof(float));
        CHECK(same_bits(got, format), "scale 0x%02x: 0x%08x, format says 0x%08x", i, bits[0], bits[1]);
        CHECK(same_bits(got, legacy), "scale 0x%02x: 0x%08x, the old body gave 0x%08x", i, bits[0], bits[2]);
    }
}

static int checks_codes    = 0;
static int generator_fails = 0;

static void check_codes(uint32_t q4) {
    const sycl::int2 table = get_int_from_table_16((int) q4, kvalues_mxfp4);
    const sycl::int2 arith = get_int_from_mxfp4((int) q4);
    checks_codes++;
    CHECK(table.x() == arith.x() && table.y() == arith.y(),
          "codes 0x%08x: table (0x%08x, 0x%08x) arithmetic (0x%08x, 0x%08x)", q4, (uint32_t) table.x(),
          (uint32_t) table.y(), (uint32_t) arith.x(), (uint32_t) arith.y());
}

static void check_codes_all() {
    const uint32_t others[3] = { 0x00000000u, 0xffffffffu, 0x5ac3963cu };
    for (int byte = 0; byte < 4; byte++) {
        const uint32_t mask = 0xffu << (8 * byte);
        for (uint32_t other : others) {
            for (uint32_t v = 0; v < 256; v++) {
                check_codes((other & ~mask) | (v << (8 * byte)));
            }
        }
    }
    for (uint32_t high = 0; high < 16; high++) {
        for (uint32_t low = 0; low < 65536; low++) {
            // Each byte: the fixed high nibble over its own low-nibble code.
            uint32_t q4 = 0x01010101u * (high << 4);
            for (int byte = 0; byte < 4; byte++) {
                q4 |= ((low >> (4 * byte)) & 0xfu) << (8 * byte);
            }
            // The sweep is only as wide as its generator: check the input before using it.
            bool generated = true;
            for (int byte = 0; byte < 4; byte++) {
                const uint32_t b = (q4 >> (8 * byte)) & 0xffu;
                generated        = generated && (b >> 4) == high && (b & 0xfu) == ((low >> (4 * byte)) & 0xfu);
            }
            if (!generated) {
                if (generator_fails++ == 0) {
                    std::fprintf(stderr, "FAIL: generator: high %u codes 0x%04x gave 0x%08x\n", high, low, q4);
                }
                continue;
            }
            check_codes(q4);
        }
    }
    CHECK(generator_fails == 0, "generator: %d inputs do not carry the nibbles the sweep claims", generator_fails);
}

// One NVFP4 block (4-byte aligned, as the kernels read qs as ints) and the two q8_1 blocks it pairs with.
struct functor_case {
    alignas(4) block_nvfp4 w;
    block_q8_1 a[QK_NVFP4 / QK8_1];
};

// Weight k of sub-block s sits in byte 8*s + k%8 of qs, low nibble for k < 8, high nibble after.
static int code_of(const block_nvfp4 & w, int s, int k) {
    const uint8_t byte = w.qs[s * (QK_NVFP4_SUB / 2) + k % (QK_NVFP4_SUB / 2)];
    return k < QK_NVFP4_SUB / 2 ? (byte & 0xf) : (byte >> 4);
}

// The sum over the sub-blocks one iqs covers: iqs 0 takes sub-blocks 0 and 1, iqs 4 takes 2 and 3.
// Weight k of sub-block s multiplies activation 16*s + k, which is in q8_1 block s/2.
static double oracle_dot(const functor_case & c, int iqs, double * mag) {
    double sum = 0.0;
    *mag       = 0.0;
    for (int s = iqs / 2; s < iqs / 2 + 2; s++) {
        const block_q8_1 & y  = c.a[s / 2];
        const double       dy = (float) y.ds[0];
        const double       dw = oracle_ue4m3(c.w.d[s]);
        for (int k = 0; k < QK_NVFP4_SUB; k++) {
            const double t = dw * k_e2m1[code_of(c.w, s, k)] * dy * y.qs[(s % 2) * QK_NVFP4_SUB + k];
            sum += t;
            *mag += std::fabs(t);
        }
    }
    return sum;
}

static void check_functor() {
    static_assert(QI_NVFP4 / VDR_NVFP4_Q8_1_MMVQ == 2, "the kernels pass iqs 0 and 4 only while QI/VDR is 2");
    std::mt19937 rng(0x4f54);
    int          cases = 0;
    for (int round = 0; round < 2048; round++) {
        functor_case c{};
        for (int s = 0; s < 4; s++) {
            // Over the rounds each sub-block position takes every scale byte.
            c.w.d[s] = (uint8_t) ((round + 64 * s) & 0xff);
        }
        for (auto & q : c.w.qs) {
            q = (uint8_t) rng();
        }
        for (auto & b : c.a) {
            const float d   = (float) ((int) (rng() % 1000) + 1) / 4096.0f;
            int         sum = 0;
            for (auto & v : b.qs) {
                v = (int8_t) ((int) (rng() % 255) - 127);
                sum += v;
            }
            b.ds = sycl::half2(sycl::half(d), sycl::half(d * sum));
        }
        for (int iqs = 0; iqs < QI_NVFP4; iqs += VDR_NVFP4_Q8_1_MMVQ) {
            double       mag  = 0.0;
            const double want = oracle_dot(c, iqs, &mag);
            const float  got  = vec_dot_nvfp4_q8_1(&c.w, c.a, iqs);
            const double err  = std::fabs((double) got - want);
            cases++;
            CHECK(err <= 1e-5 * mag + 1e-30, "functor round %d iqs %d scales %02x %02x %02x %02x: got %g want %g",
                  round, iqs, c.w.d[0], c.w.d[1], c.w.d[2], c.w.d[3], got, want);
        }
    }
    std::printf("functor: %d cases against the oracle\n", cases);
}

int main() {
    check_codes_all();
    std::printf("codes: %d inputs compared\n", checks_codes);
    check_scale_all();
    std::printf("scale: 256 bytes compared\n");
    check_functor();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
