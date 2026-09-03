// Host-only unit test for the Q8_0 dense-weight coalesced/SOA tile-alignment
// predicate (ggml/src/ggml-sycl/q8-dense-layout-rule.hpp) used by
// ggml_sycl_adjust_layout_for_tensor (ggml-sycl.cpp) and
// planner_default_device_layout (unified-cache.cpp) to decide between
// GGML_LAYOUT_COALESCED and GGML_LAYOUT_SOA for dense Q8_0 ATTENTION/FFN/
// EMBEDDING/OUTPUT_WEIGHT projections (llama.cpp-pktr).
//
// What is checked:
//   1. Shapes measured in the ticket: gemma4-E4B's K=2560 head (80 blocks/row,
//      NOT a multiple of the 32-block coalesced tile) resolves NOT aligned;
//      Mistral's K=4096/14336, gemma4's down-proj K=10240 and q/k/v K=2048
//      (all exact multiples of 32 blocks) resolve aligned.
//   2. Degenerate inputs: ne00 <= 0 and a ne00 not itself a multiple of QK8_0
//      (32) both resolve NOT aligned.
//   3. Exhaustive sweep over every blocks_per_row from 0 to 256 against an
//      independent reference implementation, so no single hand-picked shape
//      can hide an off-by-one.
//
// Pure C++: no SYCL, no ggml link. Exit 1 on any failure.

#include "ggml-sycl/q8-dense-layout-rule.hpp"

#include <cstdio>

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

// Independent reference: written with plain arithmetic, no helper calls, so
// it cannot share a bug with the implementation under test.
static bool reference_tile_aligned(int64_t ne00) {
    if (ne00 <= 0) {
        return false;
    }
    if (ne00 % 32 != 0) {
        return false;
    }
    const int64_t blocks_per_row = ne00 / 32;
    return (blocks_per_row % 32) == 0;
}

static void check_shape(int64_t ne00, bool expect_aligned, const char * label) {
    const bool got = ggml_sycl_q8_0_coalesced_tile_aligned(ne00);
    CHECK(got == expect_aligned, "%s: ne00=%lld expected aligned=%d, got %d", label, (long long) ne00,
          (int) expect_aligned, (int) got);
    CHECK(got == reference_tile_aligned(ne00), "%s: ne00=%lld disagrees with the independent reference (got %d)", label,
          (long long) ne00, (int) got);
}

int main() {
    // Ticket shapes: the one non-aligned head, and every aligned shape it
    // must NOT regress.
    check_shape(2560, false, "gemma4-E4B gate/up/q/k/v/o (K=2560, 80 blocks/row)");
    check_shape(4096, true, "Mistral 7B attn/ffn (K=4096, 128 blocks/row)");
    check_shape(14336, true, "Mistral 7B ffn (K=14336, 448 blocks/row)");
    check_shape(10240, true, "gemma4-E4B down-proj (K=10240, 320 blocks/row)");
    check_shape(2048, true, "gemma4-E4B q/k/v (K=2048, 64 blocks/row)");

    // Degenerate inputs.
    check_shape(0, false, "ne00=0");
    check_shape(-32, false, "negative ne00");
    check_shape(31, false, "ne00 not a multiple of QK8_0 (31)");
    check_shape(33, false, "ne00 not a multiple of QK8_0 (33)");
    check_shape(2559, false, "ne00 one below an aligned block-count boundary");
    check_shape(32, false, "1 block/row is not a multiple of the 32-block tile");
    check_shape(1024, true, "smallest aligned shape: 32 blocks/row (1 full tile)");

    // Exhaustive sweep: every blocks_per_row from 0 to 256 (8192 columns),
    // both as an exact multiple of QK8_0 and (for a few) with a remainder.
    for (int64_t bpr = 0; bpr <= 256; ++bpr) {
        const int64_t ne00 = bpr * 32;
        check_shape(ne00, reference_tile_aligned(ne00), "sweep (exact block boundary)");
        if (bpr > 0) {
            check_shape(ne00 - 1, false, "sweep (one below block boundary, never a QK8_0 multiple)");
        }
    }

    if (failures) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: q8_0 dense-weight coalesced/SOA tile-alignment predicate\n");
    return 0;
}
