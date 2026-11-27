// The layout of the BF16 grouped prompt arm of MUL_MAT_ID (moe-bf16-grouped-plan.hpp, Inkling 2 Mini plan Task 24).
// No device, no backend.
//
// What it pins: the plan routes every (slot, token) of the ids to exactly one packed row inside its expert's slot;
// the active experts come in ascending id with their slots contiguous and each rounded up to the row alignment; an
// expert no token selects has no slot; an id outside [0, n_experts) is refused, so the arm declines and the direct
// kernel writes that slot; a token whose picks repeat an expert gives that expert one row per pick; and, for picks that
// are distinct experts (what MUL_MAT_ID routing gives), the packed total stays within the ring's planned rows
// (pp_moe_onednn_packed_rows_at) and the distinct GEMM widths, which the oneDNN primitive cache is keyed on, stay at
// most ceil(n_tokens / align): at a 512-token micro-batch at most 8 per expert shape over every node, so at most 16
// primitives for Inkling's two shapes. Repeated picks can exceed that width bound, and one case shows it.

#include "moe-bf16-grouped-plan.hpp"
#include "moe-scratch-admission.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

using ggml_sycl::moe_bf16_grouped_expert;
using ggml_sycl::moe_bf16_grouped_plan;
using ggml_sycl::moe_bf16_grouped_plan_build;
using ggml_sycl::moe_bf16_grouped_plan_status;
using ggml_sycl::moe_bf16_grouped_plan_status_name;
using ggml_sycl::moe_bf16_grouped_row;
using ggml_sycl::moe_bf16_grouped_row_align;
using ggml_sycl::pp_moe_onednn_packed_rows_at;

static int g_failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

static void check_status(moe_bf16_grouped_plan_status got, moe_bf16_grouped_plan_status want, const char * what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s (status %s, want %s)\n", what, moe_bf16_grouped_plan_status_name(got),
                     moe_bf16_grouped_plan_status_name(want));
        ++g_failures;
    }
}

static size_t align_rows(size_t rows) {
    return (rows + moe_bf16_grouped_row_align - 1) / moe_bf16_grouped_row_align * moe_bf16_grouped_row_align;
}

static size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// Top-k distinct experts per token from `pool` (every expert when it is empty), drawn uniformly.
static std::vector<int32_t> route(std::mt19937 &               rng,
                                  size_t                       n_tokens,
                                  size_t                       n_slots,
                                  size_t                       n_experts,
                                  const std::vector<int32_t> & pool_in = {}) {
    std::vector<int32_t> pool = pool_in;
    if (pool.empty()) {
        pool.resize(n_experts);
        std::iota(pool.begin(), pool.end(), 0);
    }
    std::vector<int32_t> ids(n_tokens * n_slots);
    for (size_t t = 0; t < n_tokens; ++t) {
        std::shuffle(pool.begin(), pool.end(), rng);
        for (size_t s = 0; s < n_slots; ++s) {
            ids[t * n_slots + s] = pool[s];
        }
    }
    return ids;
}

// Everything a plan must satisfy for its ids, whatever the routing.
static bool well_formed(const std::vector<int32_t> &  ids,
                        size_t                        n_tokens,
                        size_t                        n_slots,
                        size_t                        n_experts,
                        const moe_bf16_grouped_plan & plan) {
    std::vector<size_t> want_rows(n_experts, 0);
    for (int32_t id : ids) {
        ++want_rows[static_cast<size_t>(id)];
    }
    // The experts: exactly the selected ones, ascending, each slot rounded up, the slots contiguous from row 0.
    size_t              next_base = 0;
    size_t              n_active  = 0;
    std::set<size_t>    widths;
    std::vector<size_t> expert_index(n_experts, SIZE_MAX);
    int32_t             prev = -1;
    for (size_t i = 0; i < plan.experts.size(); ++i) {
        const moe_bf16_grouped_expert & e = plan.experts[i];
        if (e.expert <= prev || e.expert < 0 || static_cast<size_t>(e.expert) >= n_experts) {
            return false;
        }
        prev = e.expert;
        if (e.rows == 0 || e.rows != want_rows[static_cast<size_t>(e.expert)] || e.gemm_rows != align_rows(e.rows) ||
            e.gemm_rows % moe_bf16_grouped_row_align != 0 || e.base_row != next_base) {
            return false;
        }
        expert_index[static_cast<size_t>(e.expert)] = i;
        next_base += e.gemm_rows;
        widths.insert(e.gemm_rows);
    }
    for (size_t e = 0; e < n_experts; ++e) {
        n_active += want_rows[e] > 0 ? 1 : 0;
    }
    if (plan.experts.size() != n_active || plan.packed_rows != next_base) {
        return false;
    }
    if (plan.gemm_widths != std::vector<size_t>(widths.begin(), widths.end())) {
        return false;
    }
    // The rows: every (slot, token) once, grouped by expert in plan order, token order inside an expert, each packed
    // row inside its expert's routed rows and every packed row used once.
    if (plan.rows.size() != ids.size()) {
        return false;
    }
    std::vector<bool> seen(ids.size(), false);
    std::vector<bool> used(plan.packed_rows, false);
    size_t            cursor = 0;
    for (const moe_bf16_grouped_expert & e : plan.experts) {
        size_t prev_flat = 0;
        for (size_t r = 0; r < e.rows; ++r, ++cursor) {
            const moe_bf16_grouped_row & row = plan.rows[cursor];
            if (row.slot < 0 || static_cast<size_t>(row.slot) >= n_slots || row.token < 0 ||
                static_cast<size_t>(row.token) >= n_tokens) {
                return false;
            }
            // ids order inside an expert: by token, then by slot.
            const size_t flat = static_cast<size_t>(row.token) * n_slots + static_cast<size_t>(row.slot);
            if (seen[flat] || ids[flat] != e.expert || (r > 0 && flat <= prev_flat)) {
                return false;
            }
            seen[flat] = true;
            prev_flat  = flat;
            if (row.packed_row < 0 || static_cast<size_t>(row.packed_row) < e.base_row ||
                static_cast<size_t>(row.packed_row) >= e.base_row + e.rows || used[row.packed_row]) {
                return false;
            }
            used[row.packed_row] = true;
        }
    }
    return cursor == plan.rows.size();
}

static void test_bad_args() {
    const int32_t         ids[2] = { 0, 1 };
    moe_bf16_grouped_plan plan;
    plan.packed_rows = 7;
    check_status(moe_bf16_grouped_plan_build(nullptr, 1, 2, 4, &plan), moe_bf16_grouped_plan_status::BAD_ARGS,
                 "null ids");
    check_status(moe_bf16_grouped_plan_build(ids, 1, 2, 4, nullptr), moe_bf16_grouped_plan_status::BAD_ARGS,
                 "null plan");
    check_status(moe_bf16_grouped_plan_build(ids, 0, 2, 4, &plan), moe_bf16_grouped_plan_status::BAD_ARGS, "no tokens");
    check_status(moe_bf16_grouped_plan_build(ids, 1, 0, 4, &plan), moe_bf16_grouped_plan_status::BAD_ARGS, "no slots");
    check_status(moe_bf16_grouped_plan_build(ids, 1, 2, 0, &plan), moe_bf16_grouped_plan_status::BAD_ARGS,
                 "no experts");
    check(plan.packed_rows == 7 && plan.experts.empty(), "a refused plan leaves the output untouched");
}

// Three tokens, two slots, four experts; expert 1 is selected by no token.
static void test_hand_case() {
    const std::vector<int32_t> ids = { 0, 2, 2, 3, 0, 2 };
    moe_bf16_grouped_plan      plan;
    check_status(moe_bf16_grouped_plan_build(ids.data(), 3, 2, 4, &plan), moe_bf16_grouped_plan_status::OK,
                 "hand case builds");
    const size_t a = moe_bf16_grouped_row_align;
    check(plan.experts.size() == 3, "hand case: three active experts, expert 1 has no slot");
    if (plan.experts.size() == 3) {
        check(plan.experts[0].expert == 0 && plan.experts[0].rows == 2 && plan.experts[0].base_row == 0,
              "hand case: expert 0 holds two rows from row 0");
        check(plan.experts[1].expert == 2 && plan.experts[1].rows == 3 && plan.experts[1].base_row == a,
              "hand case: expert 2 holds three rows from one aligned slot on");
        check(plan.experts[2].expert == 3 && plan.experts[2].rows == 1 && plan.experts[2].base_row == 2 * a,
              "hand case: expert 3 holds one row from two aligned slots on");
    }
    check(plan.packed_rows == 3 * a, "hand case: three aligned slots in all");
    check(plan.gemm_widths == std::vector<size_t>{ a }, "hand case: one GEMM width");
    // The rows in plan order: expert 0's (t0 s0, t2 s0), expert 2's (t0 s1, t1 s0, t2 s1), expert 3's (t1 s1).
    const int32_t ai         = static_cast<int32_t>(a);
    const int32_t want[6][3] = {
        { 0, 0, 0      },
        { 0, 2, 1      },
        { 1, 0, ai     },
        { 0, 1, ai + 1 },
        { 1, 2, ai + 2 },
        { 1, 1, 2 * ai },
    };
    bool rows_ok = plan.rows.size() == 6;
    for (size_t i = 0; rows_ok && i < 6; ++i) {
        rows_ok = plan.rows[i].slot == want[i][0] && plan.rows[i].token == want[i][1] &&
                  plan.rows[i].packed_row == want[i][2];
    }
    check(rows_ok, "hand case: each routed row's slot, token and packed row");
    check(well_formed(ids, 3, 2, 4, plan), "hand case is well formed");
}

// Two tokens, two slots, four experts; token 0 picks expert 2 in both slots. Each pick is its own routed row, so
// expert 2 holds three rows, two of them token 0's, in slot order.
static void test_duplicate_picks() {
    const std::vector<int32_t> ids = { 2, 2, 0, 2 };
    moe_bf16_grouped_plan      plan;
    check_status(moe_bf16_grouped_plan_build(ids.data(), 2, 2, 4, &plan), moe_bf16_grouped_plan_status::OK,
                 "duplicate picks build");
    const size_t a = moe_bf16_grouped_row_align;
    check(plan.experts.size() == 2, "duplicate picks: two active experts");
    if (plan.experts.size() == 2) {
        check(plan.experts[0].expert == 0 && plan.experts[0].rows == 1 && plan.experts[0].base_row == 0,
              "duplicate picks: expert 0 holds one row from row 0");
        check(plan.experts[1].expert == 2 && plan.experts[1].rows == 3 && plan.experts[1].base_row == a,
              "duplicate picks: expert 2 holds a row per pick, three, from one aligned slot on");
    }
    check(plan.packed_rows == 2 * a && plan.gemm_widths == std::vector<size_t>{ a },
          "duplicate picks: two aligned slots, one GEMM width");
    // The rows in plan order: expert 0's (t1 s0), expert 2's (t0 s0, t0 s1, t1 s1).
    const int32_t ai         = static_cast<int32_t>(a);
    const int32_t want[4][3] = {
        { 0, 1, 0      },
        { 0, 0, ai     },
        { 1, 0, ai + 1 },
        { 1, 1, ai + 2 },
    };
    bool rows_ok = plan.rows.size() == 4;
    for (size_t i = 0; rows_ok && i < 4; ++i) {
        rows_ok = plan.rows[i].slot == want[i][0] && plan.rows[i].token == want[i][1] &&
                  plan.rows[i].packed_row == want[i][2];
    }
    check(rows_ok, "duplicate picks: each routed row's slot, token and packed row");
    check(well_formed(ids, 2, 2, 4, plan), "duplicate picks are well formed");

    // The width bound is for distinct picks: 64 tokens whose slots 0 and 1 both pick expert 0 give it 128 rows, and
    // with the other experts' 64 rows that is two widths, past ceil(64 / align) = 1.
    const size_t         n_tokens = 64;
    std::vector<int32_t> repeated(n_tokens * 8);
    for (size_t t = 0; t < n_tokens; ++t) {
        for (size_t s = 0; s < 8; ++s) {
            repeated[t * 8 + s] = s < 2 ? 0 : static_cast<int32_t>(s - 1);
        }
    }
    moe_bf16_grouped_plan repeated_plan;
    check_status(moe_bf16_grouped_plan_build(repeated.data(), n_tokens, 8, 128, &repeated_plan),
                 moe_bf16_grouped_plan_status::OK, "a node of repeated picks builds");
    check(well_formed(repeated, n_tokens, 8, 128, repeated_plan), "a node of repeated picks is well formed");
    check(repeated_plan.gemm_widths == std::vector<size_t>{ a, 2 * a } &&
              repeated_plan.gemm_widths.size() > ceil_div(n_tokens, a),
          "repeated picks: two widths, past the distinct-picks bound");
}

// The routing the device test's two-width case uses (test-moe-bf16-grouped-device.cpp, make_hot_case): slot 0 of
// every token picks one expert and slots 1 to 7 stride the other 127, so every pick of a token is distinct. The hot
// expert holds all 512 tokens' rows and the others about 28 each: widths {64, 512}, within the ring's planned rows.
static void test_hot_expert() {
    const size_t         n_tokens = 512;
    const int32_t        hot      = 5;
    std::vector<int32_t> others;
    for (int32_t e = 0; e < 128; ++e) {
        if (e != hot) {
            others.push_back(e);
        }
    }
    std::vector<int32_t> ids(n_tokens * 8);
    for (size_t t = 0; t < n_tokens; ++t) {
        const size_t base = (t * 7 + t / 4) % others.size();
        ids[t * 8]        = hot;
        for (size_t s = 1; s < 8; ++s) {
            ids[t * 8 + s] = others[(base + (s - 1) * 18) % others.size()];
        }
    }
    moe_bf16_grouped_plan plan;
    check_status(moe_bf16_grouped_plan_build(ids.data(), n_tokens, 8, 128, &plan), moe_bf16_grouped_plan_status::OK,
                 "the hot-expert node builds");
    check(well_formed(ids, n_tokens, 8, 128, plan), "the hot-expert node is well formed");
    check(plan.gemm_widths == std::vector<size_t>{ moe_bf16_grouped_row_align, n_tokens },
          "the hot-expert node has two GEMM widths, 64 and 512");
    size_t planned = 0;
    check(
        pp_moe_onednn_packed_rows_at(static_cast<uint32_t>(n_tokens), 128, 8, &planned) && plan.packed_rows <= planned,
        "the hot-expert node fits the ring's planned rows");
}

static void test_bad_ids() {
    const std::vector<int32_t> negative = { 0, -1 };
    const std::vector<int32_t> too_big  = { 4, 1 };
    moe_bf16_grouped_plan      plan;
    plan.packed_rows = 7;
    check_status(moe_bf16_grouped_plan_build(negative.data(), 1, 2, 4, &plan), moe_bf16_grouped_plan_status::BAD_ID,
                 "an unused slot (-1) is refused");
    check_status(moe_bf16_grouped_plan_build(too_big.data(), 1, 2, 4, &plan), moe_bf16_grouped_plan_status::BAD_ID,
                 "an id of n_experts is refused");
    check(plan.packed_rows == 7 && plan.experts.empty(), "a refused id leaves the output untouched");
}

// A node whose routed rows would not fit an int32 row index is refused from its counts alone, before any id is read:
// the one id here stands for 2^31 of them.
static void test_row_overflow() {
    const int32_t         ids[1] = { 0 };
    moe_bf16_grouped_plan plan;
    plan.packed_rows      = 7;
    const size_t too_many = static_cast<size_t>(INT32_MAX) + 1;
    check_status(moe_bf16_grouped_plan_build(ids, too_many, 1, 4, &plan), moe_bf16_grouped_plan_status::ROW_OVERFLOW,
                 "2^31 tokens overflow an int32 row");
    check_status(moe_bf16_grouped_plan_build(ids, 1, too_many, 4, &plan), moe_bf16_grouped_plan_status::ROW_OVERFLOW,
                 "2^31 slots overflow an int32 row");
    check(plan.packed_rows == 7 && plan.experts.empty(), "an overflowing node leaves the output untouched");
}

// Inkling's node: 512 tokens, top-8 of 128 distinct experts.
static void test_inkling_node() {
    std::mt19937          rng(2024);
    const size_t          n_tokens = 512;
    const size_t          n_slots  = 8;
    const size_t          n_exp    = 128;
    const auto            ids      = route(rng, n_tokens, n_slots, n_exp);
    moe_bf16_grouped_plan plan;
    check_status(moe_bf16_grouped_plan_build(ids.data(), n_tokens, n_slots, n_exp, &plan),
                 moe_bf16_grouped_plan_status::OK, "Inkling node builds");
    check(well_formed(ids, n_tokens, n_slots, n_exp, plan), "Inkling node is well formed");
    size_t planned = 0;
    check(pp_moe_onednn_packed_rows_at(static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(n_exp),
                                       static_cast<uint32_t>(n_slots), &planned) &&
              plan.packed_rows <= planned,
          "Inkling node fits the ring's planned rows");
    check(plan.packed_rows <= n_tokens * n_slots + (moe_bf16_grouped_row_align - 1) * plan.experts.size(),
          "Inkling node pads at most align - 1 rows per active expert");
    check(plan.gemm_widths.size() <= ceil_div(n_tokens, moe_bf16_grouped_row_align),
          "Inkling node, distinct picks: at most ceil(n_tokens / align) GEMM widths");
}

// A node whose ids use 40 of the 128 experts: the other 88 have no slot and no GEMM.
static void test_unselected_experts() {
    std::mt19937         rng(7);
    std::vector<int32_t> subset(128);
    std::iota(subset.begin(), subset.end(), 0);
    std::shuffle(subset.begin(), subset.end(), rng);
    subset.resize(40);
    const auto            ids = route(rng, 512, 8, 128, subset);
    moe_bf16_grouped_plan plan;
    check_status(moe_bf16_grouped_plan_build(ids.data(), 512, 8, 128, &plan), moe_bf16_grouped_plan_status::OK,
                 "a 40-expert node builds");
    check(well_formed(ids, 512, 8, 128, plan), "a 40-expert node is well formed");
    check(plan.experts.size() <= 40, "a 40-expert node has at most 40 slots");
    const std::set<int32_t> allowed(subset.begin(), subset.end());
    bool                    inside = true;
    for (const moe_bf16_grouped_expert & e : plan.experts) {
        inside = inside && allowed.count(e.expert) == 1;
    }
    check(inside, "a 40-expert node gives no slot to an expert no token selects");
}

// The primitive cache bound, for distinct picks: at every token count up to a 512-token micro-batch, uniform and fully
// skewed routing, a node's distinct widths stay at most ceil(n_tokens / align). The cache is keyed on the shape and the
// width, and a layer's gate_up and down nodes route the same ids, so over every node the primitives are the (shape,
// width) pairs the two shapes' plans give: at most 8 widths per shape, 16 primitives in all.
static void test_gemm_width_bound() {
    constexpr int64_t                    k_gate_up_m = 4608;  // the two Inkling expert shapes' output widths (m)
    constexpr int64_t                    k_down_m    = 2304;
    std::set<std::pair<int64_t, size_t>> primitives;
    const auto                           key_widths = [&](const moe_bf16_grouped_plan & p) {
        for (const int64_t m : { k_gate_up_m, k_down_m }) {
            for (const size_t width : p.gemm_widths) {
                primitives.insert({ m, width });
            }
        }
    };
    std::mt19937 rng(99);
    bool         ok = true;
    for (size_t n_tokens = 1; n_tokens <= 512; ++n_tokens) {
        const size_t                       bound   = ceil_div(n_tokens, moe_bf16_grouped_row_align);
        // Uniform top-8 of 128.
        const auto                         uniform = route(rng, n_tokens, 8, 128);
        moe_bf16_grouped_plan              plan;
        const moe_bf16_grouped_plan_status uniform_status =
            moe_bf16_grouped_plan_build(uniform.data(), n_tokens, 8, 128, &plan);
        ok = ok && uniform_status == moe_bf16_grouped_plan_status::OK && plan.gemm_widths.size() <= bound;
        key_widths(plan);
        // Skewed: slot s of every token goes to expert s, so eight experts hold every row.
        std::vector<int32_t> skewed(n_tokens * 8);
        for (size_t t = 0; t < n_tokens; ++t) {
            for (size_t s = 0; s < 8; ++s) {
                skewed[t * 8 + s] = static_cast<int32_t>(s);
            }
        }
        moe_bf16_grouped_plan              skew_plan;
        const moe_bf16_grouped_plan_status skew_status =
            moe_bf16_grouped_plan_build(skewed.data(), n_tokens, 8, 128, &skew_plan);
        ok = ok && skew_status == moe_bf16_grouped_plan_status::OK && skew_plan.gemm_widths.size() <= bound &&
             well_formed(skewed, n_tokens, 8, 128, skew_plan);
        key_widths(skew_plan);
    }
    check(ok, "every token count up to 512, distinct picks: at most ceil(n_tokens / align) GEMM widths");
    const size_t per_shape = ceil_div(512, moe_bf16_grouped_row_align);
    size_t       gate_up = 0, down = 0;
    for (const auto & key : primitives) {
        gate_up += key.first == k_gate_up_m ? 1 : 0;
        down += key.first == k_down_m ? 1 : 0;
    }
    check(per_shape == 8 && gate_up <= per_shape && down <= per_shape,
          "over every node up to 512 tokens: at most ceil(512 / align) = 8 GEMM widths per shape");
    check(gate_up + down <= 16, "over every node up to 512 tokens: at most 16 primitives over Inkling's two shapes");
}

int main() {
    test_bad_args();
    test_hand_case();
    test_duplicate_picks();
    test_bad_ids();
    test_row_overflow();
    test_inkling_node();
    test_unselected_experts();
    test_hot_expert();
    test_gemm_width_bound();
    if (g_failures != 0) {
        std::fprintf(stderr, "test-moe-bf16-grouped-plan: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test-moe-bf16-grouped-plan: PASS\n");
    return 0;
}
