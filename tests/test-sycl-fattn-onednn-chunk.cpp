// The oneDNN Graph SDPA route must bound each execute's Graph scratch by splitting the op's query rows
// (llama.cpp-jwjz). One execute over a whole Mistral ubatch of 2048 rows asked for 1.5 GB at n_kv 4096 and
// 3 GB at n_kv 8192, above the 1 GiB DIRECT cap. The allocator served those requests outside the arena once
// per layer, and the depth fill overcommitted both cards.
//
// This pins the arithmetic in ggml-sycl/fattn-onednn-chunk.hpp:
//   - the per-execute scratch formula against every request size measured on hardware;
//   - the chunk plan for the shapes the failing bench produces;
//   - the plan's invariants over a grid of shapes and budgets.
//
// RED at b4f40634a, where onednn_fa_plan_row_chunks() was a stub that always returned one chunk: the
// scratch-formula checks pass there and the chunk-plan and invariant checks fail. GREEN from 637223175.
//
// Host-only: the header includes standard headers only (<algorithm>, <cstddef>, <cstdint>).
#include "ggml-sycl/fattn-onednn-chunk.hpp"

#include <cstdint>
#include <cstdio>

static int g_failures = 0;

static constexpr uint64_t kMiB = 1024ull * 1024ull;

// Half of the 1 GiB DIRECT Graph-scratch cap that both the B70 and the B50 report. Half, so a full chunk's
// buffer and the tail chunk's buffer fit under the cap together.
static constexpr size_t kHalfCap = 512ull * 1024ull * 1024ull;

// Q is f16 by the time oneDNN executes.
static constexpr size_t kF16 = 2;

static void expect_scratch(int64_t H_q, int64_t rows, int64_t ne11, uint64_t want, const char * source) {
    const uint64_t got = ggml_sycl::onednn_fa_graph_scratch_bytes(H_q, rows, ne11, kF16);
    if (got != want) {
        std::printf("FAILED: scratch(H_q=%lld rows=%lld ne11=%lld) = %llu, want %llu (%s)\n", (long long) H_q,
                    (long long) rows, (long long) ne11, (unsigned long long) got, (unsigned long long) want, source);
        g_failures++;
    }
}

static void expect_plan(int64_t      ncols,
                        int64_t      ne11,
                        int64_t      H_q,
                        size_t       budget,
                        int64_t      want_rows,
                        int64_t      want_n,
                        int64_t      want_tail,
                        const char * what) {
    const ggml_sycl::onednn_fa_row_chunks p = ggml_sycl::onednn_fa_plan_row_chunks(ncols, ne11, H_q, kF16, budget);
    if (p.rows != want_rows || p.n_chunks != want_n || p.tail_rows != want_tail) {
        std::printf(
            "FAILED: plan(ncols=%lld ne11=%lld H_q=%lld budget=%llu MiB) = {rows=%lld n=%lld tail=%lld}, want "
            "{rows=%lld n=%lld tail=%lld} (%s)\n",
            (long long) ncols, (long long) ne11, (long long) H_q, (unsigned long long) (budget / kMiB),
            (long long) p.rows, (long long) p.n_chunks, (long long) p.tail_rows, (long long) want_rows,
            (long long) want_n, (long long) want_tail, what);
        g_failures++;
    }
}

// Every plan must cover the op's rows exactly, keep the last chunk in (0, rows], fit the budget whenever
// one row does, use the fewest chunks that fit, and stay balanced.
static void check_invariants(int64_t ncols, int64_t ne11, int64_t H_q, size_t budget) {
    const ggml_sycl::onednn_fa_row_chunks p = ggml_sycl::onednn_fa_plan_row_chunks(ncols, ne11, H_q, kF16, budget);

    const uint64_t per_row    = ggml_sycl::onednn_fa_graph_scratch_bytes(H_q, 1, ne11, kF16);
    const uint64_t whole      = ggml_sycl::onednn_fa_graph_scratch_bytes(H_q, ncols, ne11, kF16);
    const uint64_t chunk_peak = ggml_sycl::onednn_fa_graph_scratch_bytes(H_q, p.rows, ne11, kF16);

    const char * broken = nullptr;
    if (p.rows < 1 || p.n_chunks < 1) {
        broken = "no chunk";
    } else if ((p.n_chunks - 1) * p.rows + p.tail_rows != ncols) {
        broken = "chunks do not cover the rows exactly";
    } else if (p.tail_rows < 1 || p.tail_rows > p.rows) {
        broken = "last chunk outside (0, rows]";
    } else if (budget != 0 && per_row <= budget && chunk_peak > budget) {
        broken = "a chunk exceeds the budget although one row fits";
    } else if ((budget == 0 || whole <= budget) && p.n_chunks != 1) {
        broken = "an op that fits the budget was split";
    } else if (budget != 0 && per_row <= budget) {
        const int64_t rows_max = (int64_t) (budget / per_row);
        const int64_t fewest   = (ncols + rows_max - 1) / rows_max;
        if (p.n_chunks != fewest) {
            broken = "more chunks than the budget needs";
        }
    }
    if (broken == nullptr && p.rows - p.tail_rows >= p.n_chunks) {
        broken = "chunks are not balanced (rows - tail >= n_chunks)";
    }
    if (broken != nullptr) {
        std::printf("FAILED: plan(ncols=%lld ne11=%lld H_q=%lld budget=%llu B) = {rows=%lld n=%lld tail=%lld}: %s\n",
                    (long long) ncols, (long long) ne11, (long long) H_q, (unsigned long long) budget,
                    (long long) p.rows, (long long) p.n_chunks, (long long) p.tail_rows, broken);
        g_failures++;
    }
}

int main() {
    // Request sizes measured on hardware, Mistral 7B (H_q = 32). The jwjz captures read the
    // "did not fit the ONEDNN zone" and "exceeds the cap by itself" WARN lines at ubatch 2048;
    // the 0oxf rows are the five sizes onednn_graph_scratch_zone_floor_bytes_swa() was fit to.
    expect_scratch(32, 2048, 2048, 768 * kMiB, "jwjz c4, n_kv 2048");
    expect_scratch(32, 2048, 4096, 1536 * kMiB, "jwjz c4/c7, n_kv 4096");
    expect_scratch(32, 512, 512, 48 * kMiB, "0oxf");
    expect_scratch(32, 512, 2048, 192 * kMiB, "0oxf");
    expect_scratch(32, 512, 8192, 768 * kMiB, "0oxf");
    expect_scratch(32, 256, 2048, 96 * kMiB, "0oxf");
    expect_scratch(32, 128, 2048, 48 * kMiB, "0oxf");
    expect_scratch(0, 2048, 2048, 0, "no heads");
    expect_scratch(32, 0, 2048, 0, "no rows");
    expect_scratch(32, 2048, 0, 0, "no keys");

    // The depth fill the failing bench runs: four ubatches of 2048 rows at n_kv 2048..8192.
    expect_plan(2048, 2048, 32, kHalfCap, 1024, 2, 1024, "jwjz depth fill, n_kv 2048");
    expect_plan(2048, 4096, 32, kHalfCap, 512, 4, 512, "jwjz depth fill, n_kv 4096");
    expect_plan(2048, 6144, 32, kHalfCap, 410, 5, 408, "jwjz depth fill, n_kv 6144");
    expect_plan(2048, 8192, 32, kHalfCap, 293, 7, 290, "jwjz depth fill, n_kv 8192");

    // A shape that already fits keeps its single execute.
    expect_plan(512, 512, 32, kHalfCap, 512, 1, 512, "pp512, fits");
    // A zero budget is no bound.
    expect_plan(2048, 8192, 32, 0, 2048, 1, 2048, "zero budget");
    // An op with no rows has no chunks.
    expect_plan(0, 8192, 32, kHalfCap, 0, 0, 0, "no rows");
    // One row alone exceeds the budget (64 heads x 1M keys = 384 MiB a row): one row per chunk.
    expect_plan(100, 1 << 20, 64, 64 * kMiB, 1, 100, 1, "one row exceeds the budget");

    const int64_t ncols_grid[]  = { 1, 7, 8, 9, 100, 512, 1000, 2048, 4096, 8192 };
    const int64_t ne11_grid[]   = { 256, 2048, 4096, 6144, 8192, 32768, 131072, 1 << 20 };
    const int64_t heads_grid[]  = { 1, 8, 32, 64 };
    const size_t  budget_grid[] = { 0, 1, 4096, 1 * kMiB, 64 * kMiB, kHalfCap, 1024 * kMiB };
    for (int64_t ncols : ncols_grid) {
        for (int64_t ne11 : ne11_grid) {
            for (int64_t H_q : heads_grid) {
                for (size_t budget : budget_grid) {
                    check_invariants(ncols, ne11, H_q, budget);
                }
            }
        }
    }

    if (g_failures != 0) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
