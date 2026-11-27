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
using ggml_sycl::moe_pp_gemm_row_align;
using ggml_sycl::moe_pp_pack_gemm_groups;
using ggml_sycl::moe_pp_pack_row_table;
using ggml_sycl::moe_pp_packed_layout;
using ggml_sycl::moe_pp_packed_row;

static int g_failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

static size_t align_rows(size_t rows) {
    return (rows + moe_pp_gemm_row_align - 1) & ~(moe_pp_gemm_row_align - 1);
}

// What every layout must satisfy, whatever the routing: the groups cover the experts in order, each expert's slot
// holds its rows, no two slots overlap, the strips are contiguous, and the total is the sum of the slots. Each group
// also ends exactly where the split rule says, at the first expert with under half the head's rows: the split rule
// decides which GEMM shapes get compiled, so a rule that splits more eagerly than the executor's must fail here.
static bool well_formed(const std::vector<size_t> & rows, const moe_pp_packed_layout & layout) {
    if (layout.groups.empty() || layout.slot_base_row.size() != rows.size()) {
        return false;
    }
    size_t next_expert = 0;
    size_t next_row    = 0;
    for (const moe_pp_gemm_group & g : layout.groups) {
        if (g.begin != next_expert || g.end <= g.begin || g.end > rows.size() || g.base_row != next_row ||
            g.n_rows % moe_pp_gemm_row_align != 0 || g.n_rows != align_rows(rows[g.begin])) {
            return false;
        }
        if (g.end != rows.size() && rows[g.end] * 2 >= rows[g.begin]) {
            return false;
        }
        for (size_t i = g.begin; i < g.end; ++i) {
            if (layout.slot_base_row[i] != g.base_row + (i - g.begin) * g.n_rows || rows[i] > g.n_rows ||
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
    check(layout.slot_base_row[511] == 511 * 64, "the last expert's slot is the last 64 rows");
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
    check(layout.slot_base_row == want_base, "each expert's base row");
    check(layout.packed_rows == 1344, "a hot expert needs 1344 rows");
    check(layout.packed_rows < align_rows(rows[0]) * rows.size(), "which is under the 6144-row rectangle");
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
// plus one alignment less a row per expert, which is what the planner sizes the ring from.
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
        if (layout.packed_rows > align_rows(rows[0]) * count ||
            layout.packed_rows > 2 * sum + (moe_pp_gemm_row_align - 1) * count) {
            check(false, "a random routing stays under the rectangle and under 2 * rows + (align - 1) * experts");
            return;
        }
    }
}

// The header returns rows, and the caller forms byte offsets by multiplying by the row width, as this test does. 40
// experts of 65536 rows of 2560 half floats put the last expert's byte offset past 4 GiB. The row counts fit in 32
// bits, so the runtime checks alone would pass with a 32-bit field; the static_asserts are what pin every row field
// to size_t, so that a caller's product is formed in size_t and cannot wrap.
static void test_offsets_past_4gib() {
    static_assert(std::is_same<decltype(moe_pp_packed_layout::packed_rows), size_t>::value, "rows are size_t");
    static_assert(std::is_same<decltype(moe_pp_packed_layout::slot_base_row)::value_type, size_t>::value,
                  "slot base rows are size_t");
    static_assert(std::is_same<decltype(moe_pp_gemm_group::base_row), size_t>::value, "group base rows are size_t");
    static_assert(std::is_same<decltype(moe_pp_gemm_group::n_rows), size_t>::value, "group slot rows are size_t");
    const std::vector<size_t> rows(40, 65536);
    moe_pp_packed_layout      layout;
    check(moe_pp_pack_gemm_groups(rows, &layout), "40 full experts pack");
    const size_t row_bytes = 2560 * sizeof(uint16_t);
    check(layout.slot_base_row[39] == size_t(39) * 65536, "the last expert's base row");
    check(layout.slot_base_row[39] * row_bytes == 13086228480ull, "its byte offset is exact past 4 GiB");
    check(layout.packed_rows * row_bytes == 13421772800ull, "the strip's size is exact past 4 GiB");
}

static void test_rejections() {
    const size_t              max_size      = std::numeric_limits<size_t>::max();
    // 2^63 and 2^62 on a 64-bit size_t.
    const size_t              half          = max_size / 2 + 1;
    const size_t              quarter       = half / 2;
    const std::vector<size_t> sentinel_base = { 11, 13 };
    moe_pp_packed_layout      layout;
    layout.packed_rows   = 7;
    layout.slot_base_row = sentinel_base;
    check(!moe_pp_pack_gemm_groups({}, &layout), "no experts is refused");
    check(!moe_pp_pack_gemm_groups({ 4, 0 }, &layout), "an expert with no rows is refused");
    check(!moe_pp_pack_gemm_groups({ 4, 5 }, &layout), "an unsorted input is refused");
    check(!moe_pp_pack_gemm_groups({ max_size }, &layout), "a row count that cannot be rounded up is refused");
    check(!moe_pp_pack_gemm_groups({ max_size / 2, max_size / 2, max_size / 2 }, &layout),
          "a total that overflows inside one group is refused");
    // Two groups: the first group ends just under half of size_t, so the second group's three slots overflow only when
    // counted from there. A check that forgets the rows already laid out accepts this and wraps packed_rows.
    check(!moe_pp_pack_gemm_groups({ half - 64, quarter - 64, quarter - 64, quarter - 64 }, &layout),
          "a total that overflows across groups is refused");
    check(!moe_pp_pack_gemm_groups({ 4 }, nullptr), "a null output is refused");
    check(layout.packed_rows == 7 && layout.groups.empty() && layout.slot_base_row == sentinel_base,
          "a refusal leaves the output untouched");
}

// The row table of three experts whose row-map spans are in expert-id order, not size order: the second-largest
// expert's rows come first in the map, the smallest's at its start. Every routed row gets one entry, in its expert's
// slot, and no entry lands on another entry's row or in a slot's padding.
static void test_row_table_three_experts() {
    const std::vector<size_t>                    rows  = { 40, 30, 3 };
    const std::vector<std::pair<size_t, size_t>> spans = {
        { 10, 50 },
        { 50, 80 },
        { 0,  3  },
    };
    moe_pp_packed_layout layout;
    check(moe_pp_pack_gemm_groups(rows, &layout), "three experts pack");
    std::vector<moe_pp_packed_row> table;
    check(moe_pp_pack_row_table(spans, layout, &table), "three experts build a row table");
    check(table.size() == 73, "the table has one entry per routed row");
    if (table.size() != 73) {
        return;
    }
    check(table[0].map_row == 10 && table[0].packed_row == 0, "entry 0 is the first expert's first row");
    check(table[39].map_row == 49 && table[39].packed_row == 39, "entry 39 is the first expert's last row");
    check(table[40].map_row == 50 && table[40].packed_row == 64, "entry 40 starts the second expert's slot at 64");
    check(table[70].map_row == 0 && table[70].packed_row == 128 && table[71].map_row == 1 &&
              table[71].packed_row == 129 && table[72].map_row == 2 && table[72].packed_row == 130,
          "the third expert's group starts at 128");
    std::vector<bool> seen(layout.packed_rows, false);
    bool              unique = true;
    for (const moe_pp_packed_row & e : table) {
        if (e.packed_row < 0 || static_cast<size_t>(e.packed_row) >= seen.size() || seen[e.packed_row]) {
            unique = false;
            break;
        }
        seen[e.packed_row] = true;
    }
    check(unique, "no packed row appears twice");
    bool   outside_padding = true;
    size_t at              = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        const size_t n = spans[i].second - spans[i].first;
        for (size_t r = 0; r < n; ++r, ++at) {
            const size_t packed = static_cast<size_t>(table[at].packed_row);
            if (packed < layout.slot_base_row[i] || packed >= layout.slot_base_row[i] + n) {
                outside_padding = false;
            }
        }
    }
    check(outside_padding, "no entry lies in an expert's padding");
}

// One expert of one row: one entry, the row-map row to packed row 0.
static void test_row_table_one_row() {
    moe_pp_packed_layout layout;
    check(moe_pp_pack_gemm_groups({ 1 }, &layout), "one row packs");
    const std::vector<std::pair<size_t, size_t>> spans = { std::make_pair(size_t(5), size_t(6)) };
    std::vector<moe_pp_packed_row>               table;
    check(moe_pp_pack_row_table(spans, layout, &table), "one row builds a row table");
    check(table.size() == 1 && table[0].map_row == 5 && table[0].packed_row == 0, "one row is entry {5, 0}");
}

static void test_row_table_rejections() {
    static_assert(sizeof(moe_pp_packed_row) == 2 * sizeof(int32_t), "the device reads the table as two ints");
    moe_pp_packed_layout two;
    check(moe_pp_pack_gemm_groups({ 4, 3 }, &two), "two experts pack");
    const moe_pp_packed_row        sentinel = { -7, -9 };
    std::vector<moe_pp_packed_row> table    = { sentinel };
    const auto                     intact   = [&]() {
        return table.size() == 1 && table[0].map_row == sentinel.map_row && table[0].packed_row == sentinel.packed_row;
    };
    using spans_t                                 = std::vector<std::pair<size_t, size_t>>;
    const std::pair<size_t, size_t> a             = { 0, 4 };
    const std::pair<size_t, size_t> b             = { 4, 7 };
    const size_t                    past_i32      = static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1;
    const spans_t                   empty_span    = { a, std::make_pair(size_t(4), size_t(4)) };
    const spans_t                   inverted_span = { a, std::make_pair(size_t(7), size_t(4)) };
    const spans_t                   three_spans   = { a, b, std::make_pair(size_t(7), size_t(8)) };
    const spans_t                   past_i32_span = { a, std::make_pair(past_i32 - 3, past_i32) };

    check(!moe_pp_pack_row_table({ a, b }, two, nullptr), "a null output is refused");
    check(!moe_pp_pack_row_table({}, two, &table) && intact(), "empty spans are refused");
    check(!moe_pp_pack_row_table({ a }, two, &table) && intact(), "a size mismatch is refused");
    check(!moe_pp_pack_row_table(empty_span, two, &table) && intact(), "an empty span is refused");
    check(!moe_pp_pack_row_table(inverted_span, two, &table) && intact(), "an inverted span is refused");

    // A hand-built layout whose one group's slots hold 2 rows, under the 4 the first span holds.
    moe_pp_packed_layout short_slot;
    short_slot.groups.push_back(moe_pp_gemm_group{ 0, 2, 2, 0 });
    short_slot.slot_base_row = { 0, 2 };
    short_slot.packed_rows   = 4;
    check(!moe_pp_pack_row_table({ a, b }, short_slot, &table) && intact(), "a span longer than its slot is refused");

    // Groups that skip position 1: the second group starts at 2.
    moe_pp_packed_layout skip;
    skip.groups.push_back(moe_pp_gemm_group{ 0, 1, 64, 0 });
    skip.groups.push_back(moe_pp_gemm_group{ 2, 3, 64, 64 });
    skip.slot_base_row = { 0, 64, 128 };
    skip.packed_rows   = 192;
    check(!moe_pp_pack_row_table(three_spans, skip, &table) && intact(), "groups that skip a position");

    // Groups that stop short: the one group covers position 0 of two.
    moe_pp_packed_layout short_groups;
    short_groups.groups.push_back(moe_pp_gemm_group{ 0, 1, 64, 0 });
    short_groups.slot_base_row = { 0, 64 };
    short_groups.packed_rows   = 128;
    check(!moe_pp_pack_row_table({ a, b }, short_groups, &table) && intact(), "groups that stop short are refused");

    check(!moe_pp_pack_row_table(past_i32_span, two, &table) && intact(), "a span end above INT32_MAX is refused");
}

int main() {
    test_even_spread();
    test_one_hot_expert();
    test_split_boundary();
    test_random_routings();
    test_offsets_past_4gib();
    test_rejections();
    test_row_table_three_experts();
    test_row_table_one_row();
    test_row_table_rejections();
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("PASS: packed MoE prompt scratch layout\n");
    return 0;
}
