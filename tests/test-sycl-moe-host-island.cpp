// The copy plan of the prompt's CPU island (ggml-sycl/moe-host-island.hpp). SYCL-free: no device is opened.
#include "ggml-sycl/moe-host-island.hpp"

#include <cstdio>
#include <vector>

using ggml_sycl::moe_island_plan;
using ggml_sycl::moe_island_plan_build;
using ggml_sycl::moe_island_row;

static int g_failed = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                \
        }                                                              \
    } while (0)

static const size_t K_BYTES = 8 * sizeof(float);  // one activation row
static const size_t N_BYTES = 4 * sizeof(float);  // one result row

// Row of `token` read by the expert in `slot`, for a layer with `n_slots` slots per token: the activation is the
// token's row of src1, the result is row (slot, token) of dst.
static moe_island_row row_of(size_t token, size_t slot, size_t n_slots) {
    moe_island_row r;
    r.src_offset = token * K_BYTES;
    r.dst_offset = (token * n_slots + slot) * N_BYTES;
    return r;
}

int main() {
    // 1. A layer whose every expert is on the host, rows given expert by expert as the island groups them
    //    (slot-major, not destination order): one copy down, one copy back.
    {
        std::vector<moe_island_row> rows;
        for (size_t slot = 0; slot < 3; ++slot) {
            for (size_t token = 0; token < 4; ++token) {
                rows.push_back(row_of(token, slot, 3));
            }
        }
        moe_island_plan plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 4);
        CHECK(plan.gather.size() == 1);
        CHECK(plan.gather[0].src_offset == 0 && plan.gather[0].dst_offset == 0 && plan.gather[0].bytes == 4 * K_BYTES);
        CHECK(plan.scatter.size() == 1);
        CHECK(plan.scatter[0].src_offset == 0 && plan.scatter[0].dst_offset == 0 &&
              plan.scatter[0].bytes == 12 * N_BYTES);
        for (size_t i = 0; i < rows.size(); ++i) {
            CHECK(plan.act_index[i] * K_BYTES == rows[i].src_offset);  // the row reads its token's activation
            CHECK(plan.out_index[i] * N_BYTES == rows[i].dst_offset);  // and writes where dst expects it
        }
    }
    // 2. Slot 1 of every token runs on the device: the host rows are slots 0 and 2. Slot 2 of token t and slot 0
    //    of token t + 1 are adjacent in dst, so the result copies are n_tokens + 1, not 2 * n_tokens.
    {
        std::vector<moe_island_row> rows;
        for (size_t slot : { size_t(0), size_t(2) }) {
            for (size_t token = 0; token < 4; ++token) {
                rows.push_back(row_of(token, slot, 3));
            }
        }
        moe_island_plan plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 4);
        CHECK(plan.gather.size() == 1);
        CHECK(plan.scatter.size() == 5);
        size_t bytes = 0;
        for (const ggml_sycl::moe_scatter_run & run : plan.scatter) {
            bytes += run.bytes;
        }
        CHECK(bytes == rows.size() * N_BYTES);
        // The host block is dense: result row r sits at r * N_BYTES whatever its destination.
        CHECK(plan.scatter[0].src_offset == 0 && plan.scatter[1].src_offset == 1 * N_BYTES);
        CHECK(plan.scatter[1].dst_offset == 2 * N_BYTES && plan.scatter[1].bytes == 2 * N_BYTES);
    }
    // 3. Only tokens 0, 1 and 5 have a host row: three activation rows in two copies.
    {
        std::vector<moe_island_row> rows = { row_of(5, 0, 1), row_of(0, 0, 1), row_of(1, 0, 1) };
        moe_island_plan             plan;
        CHECK(moe_island_plan_build(rows, K_BYTES, N_BYTES, &plan));
        CHECK(plan.n_act_rows == 3);
        CHECK(plan.gather.size() == 2);
        CHECK(plan.gather[0].bytes == 2 * K_BYTES && plan.gather[1].src_offset == 5 * K_BYTES);
        CHECK(plan.gather[1].dst_offset == 2 * K_BYTES);
        CHECK(plan.act_index[0] == 2 && plan.act_index[1] == 0 && plan.act_index[2] == 1);
    }
    // 4. Refusals: nothing to do, two rows bound for one destination, and activation rows that overlap without
    //    being the same row.
    {
        moe_island_plan plan;
        CHECK(!moe_island_plan_build({}, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> same_dst = { row_of(0, 0, 2), row_of(1, 0, 2) };
        same_dst[1].dst_offset               = same_dst[0].dst_offset;
        CHECK(!moe_island_plan_build(same_dst, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> torn_src = { row_of(0, 0, 2), row_of(1, 0, 2) };
        torn_src[1].src_offset               = K_BYTES / 2;
        CHECK(!moe_island_plan_build(torn_src, K_BYTES, N_BYTES, &plan));
        std::vector<moe_island_row> one = { row_of(0, 0, 1) };
        CHECK(!moe_island_plan_build(one, 0, N_BYTES, &plan));
        CHECK(!moe_island_plan_build(one, K_BYTES, 0, &plan));
    }
    if (g_failed != 0) {
        std::printf("FAILED %d check(s)\n", g_failed);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
