// Host-only unit tests for the index math backing k_get_rows_q8_0_aos_pair
// (ggml/src/ggml-sycl/getrows.cpp, llama.cpp-go37). No GPU, no allocation,
// no SYCL runtime -- it calls q8_0_aos_pair_block_index() and
// q8_0_aos_pair_row_out_of_range() directly from getrows.hpp, the SAME
// functions the device kernel calls (not a re-implemented copy that could
// silently drift from the kernel).
//
// Why this exists: the previous Q8_0 AoS GET_ROWS kernel processed one
// output element per thread with a hand-rolled dequant. llama.cpp-go37
// replaced it with a 2-elements/thread kernel matching this file's own
// generic k_get_rows() pattern (used for Q4_0/Q4_1/Q5_0/Q5_1), while keeping
// the i01 row-bounds guard that k_get_rows_q8_0_aos (the original scalar
// kernel) added for correctness against out-of-range gather indices. These
// tests pin both halves: the block/quant-index split that determines which
// bytes a thread reads, and the bounds check that decides whether a thread
// reads the tensor at all.
//
// This is a HYGIENE change (dedupes the dequant path onto the shared
// primitive), not a throughput fix -- an interleaved A/B on both cards
// (tracker llama.cpp-go37, comment c-m6hd) measured no PP/TG change on either
// axis, since the CPU/OP_TIMING attribution this ticket started from was
// drain-mode wall time, not recoverable time.

#include "test-skip.h"  // LLAMA_TEST_EXIT_SKIP: the one definition of "77 means skip"
#include "ggml-sycl/getrows.hpp"

#include <cstdio>

#if !defined(GGML_USE_SYCL)
int main() {
    // 77 (ctest SKIP_RETURN_CODE), not 0: no gate was evaluated in this
    // build, so this must not read as a pass.
    std::fprintf(stderr, "SKIP: GGML SYCL not enabled; no gate was evaluated.\n");
    return LLAMA_TEST_EXIT_SKIP;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

// Every i00 the v2 kernel launches is even (thread_id * 2), so ib/iqs must
// land on the first of a pair within one QK8_0=32 block -- i.e. iqs is
// always even and in [0, 30], never landing on the last element of a block
// (which would make the paired iqs+1 read cross into the next block).
static bool test_block_index_stays_within_one_block_for_even_i00() {
    for (int64_t i00 = 0; i00 < 256; i00 += 2) {
        int ib  = -1;
        int iqs = -1;
        q8_0_aos_pair_block_index(i00, ib, iqs);
        TEST_ASSERT(ib == static_cast<int>(i00 / 32), "block index must be i00/QK8_0");
        TEST_ASSERT(iqs == static_cast<int>(i00 % 32), "quant index must be i00%QK8_0");
        TEST_ASSERT(iqs % 2 == 0, "iqs must be even for a 2-elements/thread launch");
        TEST_ASSERT(iqs + 1 < 32, "iqs+1 must stay inside the same block (no cross-block read)");
    }
    return true;
}

// gemma-4-E4B's per_layer_token_embd.weight is 10752 wide (336 Q8_0 blocks
// of 32). Pin the exact shape this ticket profiled: the last in-row pair
// must resolve to the final block, final byte pair.
static bool test_block_index_matches_gemma_per_layer_embd_row_width() {
    const int64_t ne00          = 10752;
    const int64_t last_pair_i00 = ne00 - 2;  // 10750
    int           ib            = -1;
    int           iqs           = -1;
    q8_0_aos_pair_block_index(last_pair_i00, ib, iqs);
    TEST_ASSERT(ib == 335, "10750/32 must select the last of 336 blocks");
    TEST_ASSERT(iqs == 30, "10750%32 must select the last in-block pair (30,31)");
    return true;
}

// A negative or too-large gathered row index must be flagged out of range;
// every value in [0, ne01) must not be.
static bool test_row_out_of_range_boundaries() {
    const int64_t ne01 = 262144;  // per_layer_token_embd.weight's row count
    TEST_ASSERT(q8_0_aos_pair_row_out_of_range(-1, ne01), "negative row index must be out of range");
    TEST_ASSERT(q8_0_aos_pair_row_out_of_range(ne01, ne01),
                "row index == ne01 (one past the end) must be out of range");
    TEST_ASSERT(q8_0_aos_pair_row_out_of_range(ne01 + 1000, ne01), "row index far past ne01 must be out of range");
    TEST_ASSERT(!q8_0_aos_pair_row_out_of_range(0, ne01), "row index 0 must be in range");
    TEST_ASSERT(!q8_0_aos_pair_row_out_of_range(ne01 - 1, ne01), "the last valid row index must be in range");
    return true;
}

int main() {
    bool ok = true;
    ok &= test_block_index_stays_within_one_block_for_even_i00();
    ok &= test_block_index_matches_gemma_per_layer_embd_row_width();
    ok &= test_row_out_of_range_boundaries();
    std::printf("SYCL GET_ROWS Q8_0 AoS v2 index-math tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif  // GGML_USE_SYCL
