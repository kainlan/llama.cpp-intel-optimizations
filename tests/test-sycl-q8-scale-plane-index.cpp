// Host-only unit test for the Q8_0 SOA scale-plane index mapping
// (ggml/src/ggml-sycl/q8-scale-plane.hpp) used by the oneDNN WoQ-int8 PP arm's
// scale staging kernel q8_0_soa_scale_plane_to_kbn_sycl (llama.cpp-nz1k,
// prefill L2b phase 1).
//
// What is checked, against a scalar reference that does NOT use the helpers:
//   1. For several (nrows, blocks_per_row) shapes -- including non-square and
//      non-power-of-two -- transposing through the helper exactly reproduces
//      the reference [K/32][N] plane from a synthetic [N][K/32] SOA plane.
//   2. The mapping is a bijection (every source element read exactly once).
//   3. The d-plane byte offset equals what convert.cpp's SOA reorder lays down
//      (nrows*ncols int8 quant bytes precede the d plane).
//   4. POSITIVE CONTROL: the "V5" mistake -- binding the SOA order as-is --
//      is provably NOT equal to the reference on any shape with
//      nrows != blocks_per_row, so a wrong mapping would fail this test.
//
// Pure C++: no SYCL, no device, no ggml link. Exit 1 on any failure.

#include "ggml-sycl/q8-scale-plane.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
            ++failures;                                      \
        }                                                    \
    } while (0)

// Scalar reference: the oneDNN scale plane is logical (K/32, N) with strides
// {N, 1}, i.e. element (kb, row) lives at kb*nrows + row, and it must carry
// the scale of block (row, kb), which the SOA d plane stores block-major at
// row*blocks_per_row + kb. Written with plain loops, no helper calls.
static std::vector<uint16_t> reference_kbn(const std::vector<uint16_t> & soa_d, int64_t nrows, int64_t bpr) {
    std::vector<uint16_t> out(static_cast<size_t>(nrows * bpr), 0);
    for (int64_t kb = 0; kb < bpr; ++kb) {
        for (int64_t row = 0; row < nrows; ++row) {
            out[static_cast<size_t>(kb * nrows + row)] = soa_d[static_cast<size_t>(row * bpr + kb)];
        }
    }
    return out;
}

// What the device kernel does per destination element.
static std::vector<uint16_t> kernel_kbn(const std::vector<uint16_t> & soa_d,
                                        int64_t                       nrows,
                                        int64_t                       bpr,
                                        std::vector<int> &            read_count) {
    const int64_t         n = nrows * bpr;
    std::vector<uint16_t> out(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t src = ggml_sycl_q8_0_soa_scale_src_index_for_kbn(i, nrows, bpr);
        CHECK(src >= 0 && src < n, "src index out of range: i=%lld src=%lld", (long long) i, (long long) src);
        if (src >= 0 && src < n) {
            out[static_cast<size_t>(i)] = soa_d[static_cast<size_t>(src)];
            read_count[static_cast<size_t>(src)]++;
        }
    }
    return out;
}

static void run_shape(int64_t nrows, int64_t bpr) {
    const int             failures_before = failures;
    const int64_t         n               = nrows * bpr;
    // Distinct value per element so any permutation error is visible.
    std::vector<uint16_t> soa_d(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        soa_d[static_cast<size_t>(i)] = static_cast<uint16_t>((i * 2654435761ULL) & 0xFFFF);
    }

    const std::vector<uint16_t> ref = reference_kbn(soa_d, nrows, bpr);
    std::vector<int>            reads(static_cast<size_t>(n), 0);
    const std::vector<uint16_t> got = kernel_kbn(soa_d, nrows, bpr, reads);

    int64_t mismatches = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (got[static_cast<size_t>(i)] != ref[static_cast<size_t>(i)]) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0, "nrows=%lld bpr=%lld: %lld/%lld destination elements differ from the reference",
          (long long) nrows, (long long) bpr, (long long) mismatches, (long long) n);

    int64_t not_exactly_once = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (reads[static_cast<size_t>(i)] != 1) {
            ++not_exactly_once;
        }
    }
    CHECK(not_exactly_once == 0, "nrows=%lld bpr=%lld: %lld source elements not read exactly once (not a bijection)",
          (long long) nrows, (long long) bpr, (long long) not_exactly_once);

    // The two 2-D helpers agree with the flat one and with each other.
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t kb = 0; kb < bpr; ++kb) {
            const int64_t dst = ggml_sycl_q8_0_soa_scale_kbn_index(row, kb, nrows);
            const int64_t src = ggml_sycl_q8_0_soa_scale_src_index(row, kb, bpr);
            CHECK(dst == kb * nrows + row, "kbn_index(%lld,%lld) = %lld", (long long) row, (long long) kb,
                  (long long) dst);
            CHECK(src == row * bpr + kb, "src_index(%lld,%lld) = %lld", (long long) row, (long long) kb,
                  (long long) src);
            CHECK(ggml_sycl_q8_0_soa_scale_src_index_for_kbn(dst, nrows, bpr) == src,
                  "flat mapping disagrees with the 2-D helpers at row=%lld kb=%lld", (long long) row, (long long) kb);
        }
    }

    // Offset of the d plane: nrows*ncols quant bytes, ncols = bpr*32.
    const size_t expect_off = static_cast<size_t>(nrows) * static_cast<size_t>(bpr) * 32u;
    CHECK(ggml_sycl_q8_0_soa_scale_plane_offset_bytes(nrows, bpr) == expect_off,
          "d-plane offset for nrows=%lld bpr=%lld is %zu, expected %zu", (long long) nrows, (long long) bpr,
          ggml_sycl_q8_0_soa_scale_plane_offset_bytes(nrows, bpr), expect_off);

    // POSITIVE CONTROL (probe llama.cpp-ovkn V5): handing oneDNN the SOA order
    // unchanged is a different plane whenever the matrix is a genuine 2-D
    // non-square shape (a 1xN or Nx1 plane transposes to itself). If this
    // check ever passes there, the test has lost its teeth.
    if (nrows != bpr && nrows > 1 && bpr > 1) {
        int64_t identical = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (soa_d[static_cast<size_t>(i)] == ref[static_cast<size_t>(i)]) {
                ++identical;
            }
        }
        CHECK(identical != n, "positive control void: SOA order equals the reference plane for nrows=%lld bpr=%lld",
              (long long) nrows, (long long) bpr);
    }
    std::printf("shape nrows=%lld blocks_per_row=%lld: %s\n", (long long) nrows, (long long) bpr,
                failures == failures_before ? "ok" : "FAILED");
}

int main() {
    // Mistral 7B dense shapes (K/32 = 128 for K=4096, 448 for K=14336) plus
    // small, non-square and non-power-of-two cases.
    run_shape(4096, 128);
    run_shape(14336, 128);
    run_shape(4096, 448);
    run_shape(1024, 128);
    run_shape(7, 3);
    run_shape(3, 7);
    run_shape(1, 5);               // degenerate: bijection/offset checked, control skipped
    run_shape(5, 1);
    run_shape(33, 33);             // square: bijection/offset checked, control skipped
    run_shape(2560 / 32 * 3, 80);  // gemma4-ish K=2560 (bpr=80), odd row count

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: q8_0 SOA scale-plane index mapping\n");
    return 0;
}
