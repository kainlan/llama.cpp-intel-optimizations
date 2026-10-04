// Numerics of the SYCL Q2_0 device functions against an independent oracle
// (llama.cpp-s36q phase 4, tjg8 part B).
//
// Q2_0: block_q2_0 = fp16 d + 16 B of 2-bit codes per 64 weights, byte b holding
// weights 4b..4b+3 low bits first, code c -> (c - 1) * d, i.e. {-1, 0, +1, +2}.
// The SYCL activation type is block_q8_1 (32 weights), so one Q2_0 block pairs
// with two consecutive q8_1 blocks and the MMVQ kernels call the vec_dot once per
// q8_1 chunk (QI2_0 == 2 chunks, iqs selecting the chunk).
//
// The oracle is written from the format description in double precision and does
// not call any of the code under test, nor ggml's own scalar reference, so a
// kernel and its reference cannot agree by being the same code. It checks:
//   - dequantize_q2_0 (the converter behind to_fp16 / the PP dequant arm), through
//     the same index mapping dequantize_block<QK2_0, QR2_0, ...> uses;
//   - vec_dot_q2_0_q8_1 (the MMVQ vec_dot, shared by the dense and the _id path),
//     once per (block, chunk), with weight blocks at EVERY index of an 18-byte-
//     stride array. block_q2_0 is 18 bytes, so qs sits at 2 mod 4 on every other
//     block, and a 4-byte-aligned reader there is a misaligned load.
// Runs on the OpenCL CPU device (the registration pins the selector); nothing
// here touches a GPU.

#include "dequantize.hpp"
#include "ggml-common.h"
#include "ggml.h"
#include "vecdotq.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

static int g_failed = 0;

#define CHECK(cond, ...)                                \
    do {                                                \
        if (!(cond)) {                                  \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
            std::fprintf(stderr, "\n");                 \
            g_failed++;                                 \
        }                                               \
    } while (0)

static const int N_BLOCKS = 9;  // odd and even block indices, both qs alignments

int main() {
    sycl::queue q;
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::mt19937 rng(0x2a50);

    for (int round = 0; round < 8; round++) {
        // Weights: random d, random bytes, so all four codes appear in every position.
        std::vector<block_q2_0> w(N_BLOCKS);
        for (auto & b : w) {
            const float d = (static_cast<int>(rng() % 2000) - 1000) / 256.0f;
            b.d           = sycl::half(d);
            for (auto & v : b.qs) {
                v = static_cast<uint8_t>(rng());
            }
        }
        // Activations: two q8_1 chunks per weight block, random int8 quants and scale.
        std::vector<block_q8_1> a(2 * N_BLOCKS);
        for (auto & b : a) {
            const float d = (static_cast<int>(rng() % 1000) + 1) / 4096.0f;
            b.ds          = sycl::half2(sycl::half(d), sycl::half(0.0f));
            for (auto & v : b.qs) {
                v = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
            }
        }

        block_q2_0 * dw = sycl::malloc_shared<block_q2_0>(N_BLOCKS, q);
        block_q8_1 * da = sycl::malloc_shared<block_q8_1>(2 * N_BLOCKS, q);
        float *      dy = sycl::malloc_shared<float>(N_BLOCKS * QK2_0, q);
        float *      ds = sycl::malloc_shared<float>(N_BLOCKS * 2, q);
        std::memcpy(dw, w.data(), sizeof(block_q2_0) * N_BLOCKS);
        std::memcpy(da, a.data(), sizeof(block_q8_1) * 2 * N_BLOCKS);

        // dequantize_block<QK2_0, QR2_0, dequantize_q2_0> indexing, one pair per item.
        q.parallel_for(sycl::range<1>(N_BLOCKS * QK2_0 / 2), [=](sycl::id<1> id) {
             const int64_t i        = 2 * static_cast<int64_t>(id[0]);
             const int64_t ib       = i / QK2_0;
             const int64_t iqs      = (i % QK2_0) / QR2_0;
             const int64_t iybs     = i - i % QK2_0;
             const int64_t y_offset = QR2_0 == 1 ? 1 : QK2_0 / 2;
             dfloat2       v;
             dequantize_q2_0(dw, ib, static_cast<int>(iqs), v);
             dy[iybs + iqs + 0]        = static_cast<float>(v.x());
             dy[iybs + iqs + y_offset] = static_cast<float>(v.y());
         }).wait();

        // vec_dot once per (block, chunk), as mul_mat_vec_q's lane mapping calls it.
        q.parallel_for(sycl::range<2>(N_BLOCKS, QI2_0), [=](sycl::id<2> id) {
             const int ib         = static_cast<int>(id[0]);
             const int iqs        = static_cast<int>(id[1]);
             ds[ib * QI2_0 + iqs] = vec_dot_q2_0_q8_1(dw + ib, da + 2 * ib, iqs);
         }).wait();

        for (int ib = 0; ib < N_BLOCKS; ib++) {
            const double d0 = static_cast<float>(w[ib].d);
            for (int j = 0; j < QK2_0; j++) {
                const int    code = (w[ib].qs[j / 4] >> (2 * (j % 4))) & 3;
                const double ref  = (code - 1) * d0;
                CHECK(dy[ib * QK2_0 + j] == static_cast<float>(ref),
                      "dequant round %d block %d weight %d: got %g want %g", round, ib, j, dy[ib * QK2_0 + j], ref);
            }
            for (int c = 0; c < QI2_0; c++) {
                const block_q8_1 & y   = a[2 * ib + c];
                const double       d1  = static_cast<float>(y.ds[0]);
                double             sum = 0.0;
                double             mag = 0.0;
                for (int j = 0; j < QK8_1; j++) {
                    const int    wj   = c * QK8_1 + j;
                    const int    code = (w[ib].qs[wj / 4] >> (2 * (wj % 4))) & 3;
                    const double t    = d0 * (code - 1) * d1 * y.qs[j];
                    sum += t;
                    mag += std::fabs(t);
                }
                const double err = std::fabs(ds[ib * QI2_0 + c] - sum);
                CHECK(err <= 1e-5 * mag + 1e-12, "vec_dot round %d block %d chunk %d: got %g want %g (err %g, mag %g)",
                      round, ib, c, ds[ib * QI2_0 + c], sum, err, mag);
            }
        }

        sycl::free(dw, q);
        sycl::free(da, q);
        sycl::free(dy, q);
        sycl::free(ds, q);
    }

    if (g_failed != 0) {
        std::fprintf(stderr, "%d check(s) FAILED\n", g_failed);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
