// Q8_0 x Q8_1 matvec, column-split form against the row form and a CPU reference, on the device. Loads no model.
//
// The column-split form exists for matrices with few rows and long columns (the Qwen3.8 decode profile: hc_inject,
// 4 x 10240, and the 48 x 2560 node matvecs), where the row form gives each row one sub-group and the card idles.
// This test runs the production launchers of both layouts the kernel exists for, SOA and coalesced, through the
// GGML_SYCL_PRIVATE_TESTING entry mmvq_test_q8_0_launch. That entry takes the compute-unit count the dispatcher would
// pass: 0 makes the dispatch rule keep the row form, the card's own count lets the rule pick the split. It returns the
// chunks it launched with, so a case also proves which form ran.
//
// Each case builds random Q8_0 weights in the layout's byte order, a Q8_1 SOA activation vector from random floats,
// and an optional fused addend, then checks:
//   1. the row form against a CPU reference that uses the same fp16-rounded scales (rel 1e-3, abs 1e-2, the
//      kernel-vs-reference tolerance of test-sycl-mmvq-q8-0-soa-numerics);
//   2. the split form against the row form (rel 1e-4, abs 1e-2, that test's kernel-vs-kernel tolerance). The split
//      sums each row's chunks in a different order, so it is close to the row form, not bit-equal;
//   3. the split launched with the chunks the dispatch rule names for this shape and card, and more than one;
//   4. no row past the slice was written and every row of the slice was (dst is filled with NaN first, and NaN fails
//      every comparison). Cases whose row count leaves padded rows in the last work-group (one per layout at least,
//      which the host test asserts of the table) show that those rows reach the barrier and write nothing: a write
//      from one lands in the 16 sentinel floats past the slice. The CASE line prints the padded-row count.
// The case table, with the chunks each case gets on both cards, is in test-sycl-mmvq-q8-0-column-split-cases.h.
// Every element is compared numpy-allclose style: |got - ref| <= abs + rel * |ref|.
//
// Exit 0 with a final PASS line when every case passes, 1 on any FAIL, 77 when no SYCL GPU is usable or the card has
// too few compute units for the rule to pick the split on these shapes (the test could not measure what it covers).
//
// Run (one run, it is a GPU test): ctest --test-dir build -R '^test-sycl-mmvq-q8-0-column-split$' --output-on-failure

#include "mmvq-launch-geometry.hpp"
#include "mmvq.hpp"
#include "quants.hpp"
#include "test-skip.h"
#include "test-sycl-mmvq-q8-0-column-split-cases.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

static_assert(QK8_0 == QK8_1, "one Q8_0 block lines up with one Q8_1 block");
using q8_0_soa_traits = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>::traits;

constexpr int QK               = QK8_0;
constexpr int TILE_BLOCKS      = MMVQ_COALESCED_TILE_BLOCKS;
constexpr int TILE_WORD_STRIDE = TILE_BLOCKS * 4;
constexpr int TILE_BYTES       = MMVQ_COALESCED_TILE_BYTES_Q8_0;
// The blocks one sub-group covers per step of the SOA row form (mul_mat_vec_q_reorder's blocks_per_subgroup) and of
// the coalesced row form (one tile). The dispatch rule takes these as its step.
constexpr int SOA_STEP_BLOCKS = (q8_0_soa_traits::vdr_mmvq * WARP_SIZE + q8_0_soa_traits::qi - 1) / q8_0_soa_traits::qi;
constexpr int COALESCED_STEP  = TILE_BLOCKS;
constexpr int DST_GUARD       = 16;  // sentinel floats past the slice; a padded row writing there fails

constexpr float REF_REL_TOL    = 1e-3f;
constexpr float REF_ABS_TOL    = 1e-2f;
constexpr float KERNEL_REL_TOL = 1e-4f;
constexpr float KERNEL_ABS_TOL = 1e-2f;

using test_case = mmvq_q8_0_column_split_case;

struct q8_0_matrix {
    std::vector<int8_t>     qs;  // total_rows * blocks_per_row * 32, block-major
    std::vector<sycl::half> d;   // total_rows * blocks_per_row
};

struct q8_1_vector {
    std::vector<uint8_t> bytes;  // SOA: ncols quants, then ncols / 32 half2 (d, d * sum)
    std::vector<int8_t>  qs;
    std::vector<float>   d;      // fp16-rounded, as the kernel reads it
};

q8_0_matrix make_weights(std::mt19937 & rng, int rows, int ncols) {
    std::uniform_int_distribution<int>    q(-127, 127);
    std::uniform_real_distribution<float> scale(0.005f, 0.02f);
    const int                             blocks = rows * (ncols / QK);
    q8_0_matrix                           m;
    m.qs.resize(static_cast<size_t>(blocks) * QK);
    m.d.resize(static_cast<size_t>(blocks));
    for (auto & v : m.qs) {
        v = static_cast<int8_t>(q(rng));
    }
    for (auto & v : m.d) {
        v = sycl::half(scale(rng));
    }
    return m;
}

q8_1_vector make_activations(std::mt19937 & rng, int ncols) {
    std::uniform_real_distribution<float> x(-1.0f, 1.0f);
    const int                             blocks = ncols / QK;
    q8_1_vector                           y;
    y.qs.resize(static_cast<size_t>(ncols));
    y.d.resize(static_cast<size_t>(blocks));
    // Same footprint as mmvq_q8_1_soa_size(): quants, ds, and the 32-byte tail some kernels may read past ds.
    y.bytes.assign(static_cast<size_t>(ncols) + static_cast<size_t>(blocks) * sizeof(sycl::half2) + 32, 0);
    for (int b = 0; b < blocks; ++b) {
        float vals[QK];
        float amax = 0.0f;
        for (int k = 0; k < QK; ++k) {
            vals[k] = x(rng);
            amax    = std::fmax(amax, std::fabs(vals[k]));
        }
        const float d   = amax / 127.0f;
        int         sum = 0;
        for (int k = 0; k < QK; ++k) {
            const int q         = d > 0.0f ? static_cast<int>(std::lround(vals[k] / d)) : 0;
            y.qs[b * QK + k]    = static_cast<int8_t>(q);
            y.bytes[b * QK + k] = static_cast<uint8_t>(static_cast<int8_t>(q));
            sum += q;
        }
        const sycl::half2 ds(sycl::half(d), sycl::half(d * static_cast<float>(sum)));
        std::memcpy(y.bytes.data() + ncols + static_cast<size_t>(b) * sizeof(sycl::half2), &ds, sizeof(ds));
        y.d[b] = static_cast<float>(ds[0]);
    }
    return y;
}

// SOA: every block's 32 quants in block order, then every block's fp16 scale (block_q_t<Q8_0>::get_d_offset).
std::vector<uint8_t> soa_bytes(const q8_0_matrix & m) {
    std::vector<uint8_t> out(m.qs.size() + m.d.size() * sizeof(sycl::half));
    std::memcpy(out.data(), m.qs.data(), m.qs.size());
    std::memcpy(out.data() + m.qs.size(), m.d.data(), m.d.size() * sizeof(sycl::half));
    return out;
}

// Coalesced: per row, whole 32-block tiles in which 4-byte word w of block b sits at w * 128 + b * 4 (the tail tile
// of a row whose block count is not a multiple of 32 is padded); after all rows' quants, the fp16 scales in block
// order (mul_mat_vec_q8_0_coalesced's addressing).
std::vector<uint8_t> coalesced_bytes(const q8_0_matrix & m, int rows, int ncols) {
    const int            blocks_per_row = ncols / QK;
    const int            tiles          = (blocks_per_row + TILE_BLOCKS - 1) / TILE_BLOCKS;
    const size_t         row_stride     = static_cast<size_t>(tiles) * TILE_BYTES;
    std::vector<uint8_t> out(static_cast<size_t>(rows) * row_stride + m.d.size() * sizeof(sycl::half), 0);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const int      tile = b / TILE_BLOCKS;
            const int      bit  = b % TILE_BLOCKS;
            const int8_t * src  = m.qs.data() + (static_cast<size_t>(r) * blocks_per_row + b) * QK;
            for (int w = 0; w < QK / 4; ++w) {
                uint8_t * dst = out.data() + r * row_stride + static_cast<size_t>(tile) * TILE_BYTES +
                                static_cast<size_t>(w) * TILE_WORD_STRIDE + static_cast<size_t>(bit) * 4;
                std::memcpy(dst, src + w * 4, 4);
            }
        }
    }
    std::memcpy(out.data() + static_cast<size_t>(rows) * row_stride, m.d.data(), m.d.size() * sizeof(sycl::half));
    return out;
}

std::vector<float> reference(const test_case &          tc,
                             const q8_0_matrix &        m,
                             const q8_1_vector &        y,
                             const std::vector<float> & addend) {
    const int          blocks_per_row = tc.ncols / QK;
    std::vector<float> out(static_cast<size_t>(tc.nrows));
    for (int i = 0; i < tc.nrows; ++i) {
        const int r   = tc.row_low + i;
        double    acc = 0.0;
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t blk  = static_cast<size_t>(r) * blocks_per_row + b;
            int          sumi = 0;
            for (int k = 0; k < QK; ++k) {
                sumi += static_cast<int>(m.qs[blk * QK + k]) * static_cast<int>(y.qs[b * QK + k]);
            }
            acc += static_cast<double>(static_cast<float>(m.d[blk])) * static_cast<double>(y.d[b]) * sumi;
        }
        out[i] = static_cast<float>(acc) + (tc.fused ? addend[r] : 0.0f);
    }
    return out;
}

struct compare_result {
    bool   ok         = true;
    float  max_diff   = 0.0f;
    size_t violations = 0;
};

compare_result compare(const std::vector<float> & got, const std::vector<float> & ref, float rel, float abs_tol) {
    compare_result res;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        if (!(diff <= abs_tol + rel * std::fabs(ref[i]))) {
            res.ok = false;
            ++res.violations;
        }
        // Once a NaN is seen it stays the reported maximum: an unwritten row must not hide behind a later finite one.
        if (!std::isnan(res.max_diff) && !(diff <= res.max_diff)) {
            res.max_diff = diff;
        }
    }
    return res;
}

struct device_run {
    std::vector<float> out;
    int                chunks      = 0;
    bool               guard_clean = true;
};

device_run run(sycl::queue &     q,
               const test_case & tc,
               const void *      vx,
               const void *      vy,
               const float *     addend_d,
               float *           dst_d,
               int               compute_units) {
    const size_t       n = static_cast<size_t>(tc.nrows) + DST_GUARD;
    std::vector<float> fill(n, std::numeric_limits<float>::quiet_NaN());
    const float        sentinel = -12345.0f;
    for (size_t i = tc.nrows; i < n; ++i) {
        fill[i] = sentinel;
    }
    q.memcpy(dst_d, fill.data(), n * sizeof(float)).wait();

    device_run r;
    r.chunks =
        mmvq_test_q8_0_launch(q, tc.coalesced, vx, vy, dst_d, tc.ncols, tc.nrows, tc.total_rows, tc.row_low,
                              compute_units, tc.fused ? addend_d : nullptr, tc.fused ? tc.total_rows : 0, tc.row_low);
    q.wait_and_throw();

    std::vector<float> all(n);
    q.memcpy(all.data(), dst_d, n * sizeof(float)).wait();
    for (size_t i = tc.nrows; i < n; ++i) {
        if (all[i] != sentinel) {
            r.guard_clean = false;
        }
    }
    r.out.assign(all.begin(), all.begin() + tc.nrows);
    return r;
}

}  // namespace

int main() {
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    } catch (const sycl::exception & e) {
        std::fprintf(stderr, "SKIP: no usable SYCL GPU: %s\n", e.what());
        return LLAMA_TEST_EXIT_SKIP;
    }
    const int compute_units = static_cast<int>(q.get_device().get_info<sycl::info::device::max_compute_units>());
    std::printf("device: %s compute_units=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                compute_units);

    const auto & cases = MMVQ_Q8_0_COLUMN_SPLIT_CASES;

    for (const test_case & tc : cases) {
        const int step = tc.coalesced ? COALESCED_STEP : SOA_STEP_BLOCKS;
        if (ggml_sycl::mmvq_column_split_chunks(tc.nrows, tc.ncols / QK, step, compute_units) <= 1) {
            std::fprintf(stderr, "SKIP: %d compute units keep the row form on %s; nothing to compare\n", compute_units,
                         tc.name);
            return LLAMA_TEST_EXIT_SKIP;
        }
    }

    std::mt19937 rng(20261120);
    int          failed = 0;
    int          ran    = 0;

    try {
        for (const test_case & tc : cases) {
            const q8_0_matrix          m = make_weights(rng, tc.total_rows, tc.ncols);
            const q8_1_vector          y = make_activations(rng, tc.ncols);
            const std::vector<uint8_t> x = tc.coalesced ? coalesced_bytes(m, tc.total_rows, tc.ncols) : soa_bytes(m);
            std::vector<float>         addend(static_cast<size_t>(tc.total_rows));
            std::uniform_real_distribution<float> a(-4.0f, 4.0f);
            for (auto & v : addend) {
                v = a(rng);
            }

            void *  x_d      = sycl::malloc_device(x.size(), q);
            void *  y_d      = sycl::malloc_device(y.bytes.size(), q);
            float * addend_d = sycl::malloc_device<float>(addend.size(), q);
            float * dst_d    = sycl::malloc_device<float>(static_cast<size_t>(tc.nrows) + DST_GUARD, q);
            q.memcpy(x_d, x.data(), x.size());
            q.memcpy(y_d, y.bytes.data(), y.bytes.size());
            q.memcpy(addend_d, addend.data(), addend.size() * sizeof(float));
            q.wait();

            const device_run row   = run(q, tc, x_d, y_d, addend_d, dst_d, 0);
            const device_run split = run(q, tc, x_d, y_d, addend_d, dst_d, compute_units);

            sycl::free(x_d, q);
            sycl::free(y_d, q);
            sycl::free(addend_d, q);
            sycl::free(dst_d, q);

            const std::vector<float> ref       = reference(tc, m, y, addend);
            const compare_result     row_ref   = compare(row.out, ref, REF_REL_TOL, REF_ABS_TOL);
            const compare_result     split_row = compare(split.out, row.out, KERNEL_REL_TOL, KERNEL_ABS_TOL);
            const compare_result     split_ref = compare(split.out, ref, REF_REL_TOL, REF_ABS_TOL);
            const int                want      = ggml_sycl::mmvq_column_split_chunks(
                tc.nrows, tc.ncols / QK, tc.coalesced ? COALESCED_STEP : SOA_STEP_BLOCKS, compute_units);

            std::string why;
            if (row.chunks != 1) {
                why += " row-form-did-not-run";
            }
            if (split.chunks != want || split.chunks <= 1) {
                why += " split-chunks";
            }
            if (!row_ref.ok) {
                why += " row-vs-ref";
            }
            if (!split_row.ok) {
                why += " split-vs-row";
            }
            if (!split_ref.ok) {
                why += " split-vs-ref";
            }
            if (!row.guard_clean || !split.guard_clean) {
                why += " wrote-past-slice";
            }
            const bool ok     = why.empty();
            const int  padded = ggml_sycl::mmvq_column_split_workgroups(tc.nrows, want) *
                                   ggml_sycl::mmvq_column_split_rows_per_workgroup(want) -
                               tc.nrows;
            std::printf(
                "CASE name=%s layout=%s rows=%d/%d row_low=%d ncols=%d fused=%d chunks=%d (rule %d) padded=%d "
                "row-vs-ref max_diff=%.3e viol=%zu split-vs-row max_diff=%.3e viol=%zu split-vs-ref max_diff=%.3e "
                "-> %s%s\n",
                tc.name, tc.coalesced ? "coalesced" : "soa", tc.nrows, tc.total_rows, tc.row_low, tc.ncols,
                tc.fused ? 1 : 0, split.chunks, want, padded, row_ref.max_diff, row_ref.violations, split_row.max_diff,
                split_row.violations, split_ref.max_diff, ok ? "PASS" : "FAIL:", why.c_str());
            ++ran;
            if (!ok) {
                ++failed;
            }
        }
    } catch (const sycl::exception & e) {
        std::printf("FAIL: SYCL exception: %s\n", e.what());
        return 1;
    }

    if (failed != 0) {
        std::printf("FAIL: q8_0 column-split matvec: %d of %d cases failed\n", failed, ran);
        return 1;
    }
    std::printf("PASS: q8_0 column-split matvec (%d cases)\n", ran);
    return 0;
}
