// The packed layout of the batched MoE prompt executor's scratch (moe-pp-pack.hpp). No device, no backend.

#include "moe-pp-pack.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using ggml_sycl::moe_pp_gemm_group;
using ggml_sycl::moe_pp_pack_gemm_groups;
using ggml_sycl::moe_pp_packed_layout;

static int g_failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

static size_t align_64(size_t rows) {
    return (rows + 63) & ~static_cast<size_t>(63);
}

// What every layout must satisfy, whatever the routing: the groups cover the experts in order, each expert's slot
// holds its rows, no two slots overlap, the strips are contiguous, and the total is the sum of the slots.
static bool well_formed(const std::vector<size_t> & rows, const moe_pp_packed_layout & layout) {
    if (layout.groups.empty() || layout.expert_base_row.size() != rows.size()) {
        return false;
    }
    size_t next_expert = 0;
    size_t next_row    = 0;
    for (const moe_pp_gemm_group & g : layout.groups) {
        if (g.begin != next_expert || g.end <= g.begin || g.base_row != next_row || g.n_rows % 64 != 0 ||
            g.n_rows != align_64(rows[g.begin])) {
            return false;
        }
        for (size_t i = g.begin; i < g.end; ++i) {
            if (layout.expert_base_row[i] != g.base_row + (i - g.begin) * g.n_rows || rows[i] > g.n_rows ||
                rows[i] * 2 < rows[g.begin]) {
                return false;
            }
        }
        next_expert = g.end;
        next_row    = g.base_row + (g.end - g.begin) * g.n_rows;
    }
    return next_expert == rows.size() && layout.packed_rows == next_row;
}

// 512 experts of 10 rows each: one group, 64 rows a slot. The rectangle is the same size here.
static void test_even_spread() {
    const std::vector<size_t> rows(512, 10);
    moe_pp_packed_layout      layout;
    check(moe_pp_pack_gemm_groups(rows, &layout), "an even spread packs");
    check(well_formed(rows, layout), "an even spread is well formed");
    check(layout.groups.size() == 1 && layout.groups[0].n_rows == 64, "an even spread is one group of 64-row slots");
    check(layout.packed_rows == 512 * 64, "an even spread needs 32768 rows");
    check(layout.expert_base_row[511] == 511 * 64, "the last expert's slot is the last 64 rows");
}

// One expert takes every token, the rest take a few: the rectangle pads all six to 1024 rows, the packed layout pads
// each group to its own head.
static void test_one_hot_expert() {
    const std::vector<size_t> rows = { 1024, 30, 30, 20, 4, 4 };
    moe_pp_packed_layout      layout;
    check(moe_pp_pack_gemm_groups(rows, &layout), "a hot expert packs");
    check(well_formed(rows, layout), "a hot expert is well formed");
    check(layout.groups.size() == 3, "a hot expert gives three groups");
    if (layout.groups.size() == 3) {
        const moe_pp_gemm_group & a = layout.groups[0];
        const moe_pp_gemm_group & b = layout.groups[1];
        const moe_pp_gemm_group & c = layout.groups[2];
        check(a.begin == 0 && a.end == 1 && a.n_rows == 1024 && a.base_row == 0, "group 1 is the hot expert alone");
        check(b.begin == 1 && b.end == 4 && b.n_rows == 64 && b.base_row == 1024, "group 2 is the three mid experts");
        check(c.begin == 4 && c.end == 6 && c.n_rows == 64 && c.base_row == 1216, "group 3 is the two small experts");
    }
    const std::vector<size_t> want_base = { 0, 1024, 1088, 1152, 1216, 1280 };
    check(layout.expert_base_row == want_base, "each expert's base row");
    check(layout.packed_rows == 1344, "a hot expert needs 1344 rows");
    check(layout.packed_rows < align_64(rows[0]) * rows.size(), "which is under the 6144-row rectangle");
}

// The split point: exactly half the head stays in the group, one row under half starts a new one.
static void test_split_boundary() {
    moe_pp_packed_layout layout;
    check(moe_pp_pack_gemm_groups({ 100, 50 }, &layout) && layout.groups.size() == 1, "half the head stays");
    check(moe_pp_pack_gemm_groups({ 100, 49 }, &layout) && layout.groups.size() == 2, "under half splits");
    check(moe_pp_pack_gemm_groups({ 7 }, &layout) && layout.groups.size() == 1 && layout.packed_rows == 64,
          "one expert is one 64-row slot");
}

// Random routings: the packed layout is never larger than the rectangle, and never larger than twice the routed rows
// plus 63 per expert, which is what the planner sizes the ring from.
static void test_random_routings() {
    std::mt19937 rng(20261115);
    for (int round = 0; round < 2000; ++round) {
        const size_t        count = 1 + rng() % 600;
        const size_t        cap   = 1 + rng() % 5000;
        std::vector<size_t> rows(count);
        size_t              sum = 0;
        for (size_t & r : rows) {
            // Mostly small counts with an occasional large one, like routed prompt rows.
            r = 1 + (rng() % 8 == 0 ? rng() % cap : rng() % 40);
            sum += r;
        }
        std::sort(rows.begin(), rows.end(), std::greater<size_t>());
        moe_pp_packed_layout layout;
        if (!moe_pp_pack_gemm_groups(rows, &layout) || !well_formed(rows, layout)) {
            check(false, "a random routing packs into a well-formed layout");
            return;
        }
        if (layout.packed_rows > align_64(rows[0]) * count || layout.packed_rows > 2 * sum + 63 * count) {
            check(false, "a random routing stays under the rectangle and under 2 * rows + 63 * experts");
            return;
        }
    }
}

// Offsets are size_t rows; a caller multiplies by the row width. 40 experts of 65536 rows of 2560 half floats put the
// last expert's byte offset past 4 GiB, which a 32-bit offset would wrap.
static void test_offsets_past_4gib() {
    static_assert(std::is_same<decltype(moe_pp_packed_layout::packed_rows), size_t>::value, "rows are size_t");
    const std::vector<size_t> rows(40, 65536);
    moe_pp_packed_layout      layout;
    check(moe_pp_pack_gemm_groups(rows, &layout), "40 full experts pack");
    const size_t row_bytes = 2560 * sizeof(uint16_t);
    check(layout.expert_base_row[39] == size_t(39) * 65536, "the last expert's base row");
    check(layout.expert_base_row[39] * row_bytes == 13086228480ull, "its byte offset is exact past 4 GiB");
    check(layout.packed_rows * row_bytes == 13421772800ull, "the strip's size is exact past 4 GiB");
}

static void test_rejections() {
    const size_t         max_size = std::numeric_limits<size_t>::max();
    moe_pp_packed_layout layout;
    layout.packed_rows = 7;
    check(!moe_pp_pack_gemm_groups({}, &layout), "no experts is refused");
    check(!moe_pp_pack_gemm_groups({ 4, 0 }, &layout), "an expert with no rows is refused");
    check(!moe_pp_pack_gemm_groups({ 4, 5 }, &layout), "an unsorted input is refused");
    check(!moe_pp_pack_gemm_groups({ max_size }, &layout), "a row count that cannot be rounded up is refused");
    check(!moe_pp_pack_gemm_groups({ max_size / 2, max_size / 2, max_size / 2 }, &layout),
          "a total that overflows is refused");
    check(!moe_pp_pack_gemm_groups({ 4 }, nullptr), "a null output is refused");
    check(layout.packed_rows == 7 && layout.groups.empty(), "a refusal leaves the output untouched");
}

int main() {
    test_even_spread();
    test_one_hot_expert();
    test_split_boundary();
    test_random_routings();
    test_offsets_past_4gib();
    test_rejections();
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("PASS: packed MoE prompt scratch layout\n");
    return 0;
}
