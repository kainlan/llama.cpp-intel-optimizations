// Host-side decisions of the batch-1 (decode) MUL_MAT_ID path (llama.cpp-yx28).
// ggml-sycl/moe-decode-hostpath.hpp is SYCL-free, so these run without a device.
#include "ggml-sycl/moe-decode-hostpath.hpp"

#include <cstdio>
#include <vector>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

static ggml_sycl::moe_decode_direct_request decode_request() {
    ggml_sycl::moe_decode_direct_request r;
    r.src1_tokens  = 1;
    r.ids_tokens   = 1;
    r.ids_selected = 10;
    return r;
}

static int test_direct_request() {
    using ggml_sycl::moe_decode_direct_request_admissible;
    ggml_sycl::moe_decode_direct_request r = decode_request();
    CHECK(moe_decode_direct_request_admissible(r), "a batch-1 decode op is admissible");

    r             = decode_request();
    r.src1_tokens = 2;
    CHECK(!moe_decode_direct_request_admissible(r), "a prompt batch never takes the decode direct route");
    r            = decode_request();
    r.ids_tokens = 4;
    CHECK(!moe_decode_direct_request_admissible(r), "more than one routed token is not decode");
    r              = decode_request();
    r.ids_selected = 0;
    CHECK(!moe_decode_direct_request_admissible(r), "no selected experts means nothing to dispatch");
    r                 = decode_request();
    r.graph_recording = true;
    CHECK(!moe_decode_direct_request_admissible(r), "graph recording keeps its own MoE handling");
    r                 = decode_request();
    r.layout_override = true;
    CHECK(!moe_decode_direct_request_admissible(r), "a diagnostic layout override keeps the retained route");
    r                        = decode_request();
    r.dedicated_decode_route = true;
    CHECK(!moe_decode_direct_request_admissible(r), "types with their own decode executor are not rerouted");
    return 0;
}

static int test_direct_stamp() {
    using ggml_sycl::moe_decode_direct_stamp_current;
    ggml_sycl::moe_decode_direct_stamp s{};
    CHECK(!moe_decode_direct_stamp_current(s, 1, 1, 10), "a never-evaluated tensor must be evaluated");

    ggml_sycl::moe_decode_direct_stamp_record(s, 3, 7, 10, /*layout=*/2, /*eligible=*/true);
    CHECK(moe_decode_direct_stamp_current(s, 3, 7, 10), "same generations and rows reuse the decision");
    CHECK(s.eligible && s.layout == 2, "the recorded decision and layout are kept");
    CHECK(!moe_decode_direct_stamp_current(s, 4, 7, 10), "a replan must re-evaluate residency");
    CHECK(!moe_decode_direct_stamp_current(s, 3, 8, 10), "an expert storage rewrite must re-evaluate");
    CHECK(!moe_decode_direct_stamp_current(s, 3, 7, 8), "a different selected-row count must re-evaluate");

    // An ineligible decision is cached too: a tensor with host experts is not
    // re-probed on every decode op of the same generation.
    ggml_sycl::moe_decode_direct_stamp_record(s, 3, 9, 10, /*layout=*/0, /*eligible=*/false);
    CHECK(moe_decode_direct_stamp_current(s, 3, 9, 10) && !s.eligible, "an ineligible decision is cached");

    // Settling: a refusal is a decision, a failed attempt is not.
    using ggml_sycl::moe_decode_direct_stamp_settle;
    ggml_sycl::moe_decode_direct_stamp t{};
    moe_decode_direct_stamp_settle(t, 3, 7, 10, /*layout=*/2, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE);
    CHECK(moe_decode_direct_stamp_current(t, 3, 7, 10) && t.eligible && t.layout == 2, "eligible is remembered");
    moe_decode_direct_stamp_settle(t, 3, 8, 10, /*layout=*/0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED);
    CHECK(moe_decode_direct_stamp_current(t, 3, 8, 10) && !t.eligible, "a structural refusal is remembered");
    moe_decode_direct_stamp_settle(t, 3, 9, 10, /*layout=*/2, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY);
    CHECK(!moe_decode_direct_stamp_current(t, 3, 9, 10), "a transient failure is not remembered as ineligible");
    CHECK(!moe_decode_direct_stamp_current(t, 3, 8, 10), "a transient failure voids the earlier decision");
    return 0;
}

static int test_direct_layout() {
    using ggml_sycl::moe_decode_direct_layout_from_materialized;
    int layout = -1;
    CHECK(!moe_decode_direct_layout_from_materialized({}, &layout),
          "an expert with no record on the device has no layout");
    CHECK(moe_decode_direct_layout_from_materialized({ 3 }, &layout) && layout == 3,
          "the one materialized layout is the route's layout");
    layout = -1;
    CHECK(moe_decode_direct_layout_from_materialized({ 3, 3 }, &layout) && layout == 3,
          "repeated records of one layout still name that layout");
    CHECK(!moe_decode_direct_layout_from_materialized({ 1, 3 }, &layout),
          "two kernel-readable materialized layouts name no single one");
    return 0;
}

static int test_pool_ring() {
    using ggml_sycl::moe_pool_reserve_first;
    using ggml_sycl::moe_pool_spans_disjoint;
    // A 20-entry pool serving top-10 ops alternates halves: gate, up, down.
    CHECK(moe_pool_reserve_first(0, 10, 20) == 0, "the first reservation starts at the cursor");
    CHECK(moe_pool_reserve_first(10, 10, 20) == 10, "a reservation that fits stays at the cursor");
    CHECK(moe_pool_reserve_first(0, 10, 10) == 0, "a pool of exactly one op restarts every op");
    CHECK(moe_pool_reserve_first(15, 10, 20) == 0, "a reservation past the end restarts at entry 0");
    CHECK(moe_pool_spans_disjoint(0, 10, 10, 10), "adjacent spans are disjoint");
    CHECK(!moe_pool_spans_disjoint(0, 10, 9, 10), "an overlapping span is not disjoint");
    CHECK(!moe_pool_spans_disjoint(5, 10, 0, 10), "overlap is detected in either order");
    return 0;
}

static int test_gather_runs() {
    std::vector<ggml_sycl::moe_gather_run> runs;
    // Down projection, all ten slots on the CPU: one contiguous [640, 10] block.
    std::vector<size_t>                    src;
    for (size_t i = 0; i < 10; ++i) {
        src.push_back(4096 + i * 2560);
    }
    ggml_sycl::moe_gather_runs_build(src, 7680, 2560, runs);
    CHECK(runs.size() == 1, "ten contiguous rows gather in one copy");
    CHECK(runs[0].src_offset == 4096 && runs[0].dst_offset == 7680 && runs[0].bytes == 25600,
          "the single run covers every row");

    // A hole in the source (slot 3 ran on a GPU) splits the gather.
    src = { 0, 2560, 5120, 10240, 12800 };
    ggml_sycl::moe_gather_runs_build(src, 0, 2560, runs);
    CHECK(runs.size() == 2, "a source gap starts a new run");
    CHECK(runs[1].src_offset == 10240 && runs[1].dst_offset == 7680 && runs[1].bytes == 5120,
          "the second run lands right after the first in the destination");

    // Rows out of source order never merge backwards.
    src = { 2560, 0 };
    ggml_sycl::moe_gather_runs_build(src, 0, 2560, runs);
    CHECK(runs.size() == 2, "descending sources are copied row by row");

    src.clear();
    ggml_sycl::moe_gather_runs_build(src, 0, 2560, runs);
    CHECK(runs.empty(), "no rows, no copies");
    return 0;
}

// Stand-ins for graph nodes: only their identity matters.
static const int k_gate_dst  = 0;
static const int k_up_dst    = 0;
static const int k_src1      = 0;
static const int k_next_src1 = 0;
static const int k_down_dst  = 0;

// Gate made the copy of its src1 row; up is the one sibling allowed to reuse it.
static ggml_sycl::moe_shared_act_record act_record() {
    ggml_sycl::moe_shared_act_record r;
    r.src1_tensor    = &k_src1;
    r.sibling_dst    = &k_up_dst;
    r.graph_epoch    = 2;
    r.serial         = 4;
    r.scatter_serial = 9;
    r.view_offset    = 128;
    r.bytes          = 10240;
    r.device         = 0;
    r.valid          = true;
    return r;
}

static ggml_sycl::moe_shared_act_query up_query() {
    ggml_sycl::moe_shared_act_query q;
    q.src1_tensor    = &k_src1;
    q.op_dst         = &k_up_dst;
    q.same_source    = true;
    q.graph_epoch    = 2;
    q.scatter_serial = 9;
    q.view_offset    = 128;
    q.bytes          = 10240;
    q.device         = 0;
    return q;
}

static int test_shared_activation() {
    using ggml_sycl::moe_shared_act_reusable;
    const ggml_sycl::moe_shared_act_record r = act_record();
    ggml_sycl::moe_shared_act_query        q = up_query();
    CHECK(moe_shared_act_reusable(r, q), "up reuses its sibling gate's copy of the same src1 tensor");

    // The storage location alone proves nothing: ggml-alloc puts the next
    // layer's ffn input at the same compute-buffer offset.
    q             = up_query();
    q.src1_tensor = &k_next_src1;
    CHECK(!moe_shared_act_reusable(r, q), "the same location holding a different tensor is not the copied row");
    q        = up_query();
    q.op_dst = &k_gate_dst;
    CHECK(!moe_shared_act_reusable(r, q), "the op that made the copy is not its sibling");
    q        = up_query();
    q.op_dst = &k_down_dst;
    CHECK(!moe_shared_act_reusable(r, q), "an op that is not the recorded sibling never reuses the copy");
    ggml_sycl::moe_shared_act_record unpaired = r;
    unpaired.sibling_dst                      = nullptr;
    q                                         = up_query();
    q.op_dst                                  = nullptr;
    CHECK(!moe_shared_act_reusable(unpaired, q), "a copy made by an op with no gate/up sibling is never reused");
    q             = up_query();
    q.graph_epoch = 3;
    CHECK(!moe_shared_act_reusable(r, q), "a copy from an earlier graph compute is stale");

    q             = up_query();
    q.same_source = false;
    CHECK(!moe_shared_act_reusable(r, q), "a different src1 storage needs its own copy");
    q                = up_query();
    q.scatter_serial = 10;
    CHECK(!moe_shared_act_reusable(r, q),
          "a scatter enqueued after the copy breaks the pool ordering the copy's wait provided");
    q             = up_query();
    q.view_offset = 0;
    CHECK(!moe_shared_act_reusable(r, q), "another view of the buffer is another row");
    q       = up_query();
    q.bytes = 8192;
    CHECK(!moe_shared_act_reusable(r, q), "a different row size is a different copy");
    q        = up_query();
    q.device = 1;
    CHECK(!moe_shared_act_reusable(r, q), "another device has its own staging");
    ggml_sycl::moe_shared_act_record cleared = r;
    cleared.valid                            = false;
    CHECK(!moe_shared_act_reusable(cleared, up_query()), "a cleared record is never reused");
    return 0;
}

static ggml_sycl::moe_sibling_pending_request sibling_request() {
    ggml_sycl::moe_sibling_pending_request r;
    r.pending_active     = true;
    r.sibling_slot_free  = true;
    r.reuses_activation  = true;
    r.pending_act_serial = 4;
    r.current_act_serial = 4;
    r.pending_from_pool  = true;
    r.op_from_pool       = true;
    r.pending_first      = 0;
    r.pending_count      = 10;
    r.op_first           = 10;
    r.op_count           = 10;
    r.same_row_geometry  = true;
    return r;
}

static int test_sibling_pending() {
    using ggml_sycl::moe_sibling_pending_keep;
    ggml_sycl::moe_sibling_pending_request r = sibling_request();
    CHECK(moe_sibling_pending_keep(r), "gate stays pending while up is issued into the other pool half");

    r                = sibling_request();
    r.pending_active = false;
    CHECK(!moe_sibling_pending_keep(r), "nothing pending, nothing to keep");
    r                   = sibling_request();
    r.sibling_slot_free = false;
    CHECK(!moe_sibling_pending_keep(r), "two pending jobs already fill both slots");
    r                   = sibling_request();
    r.reuses_activation = false;
    CHECK(!moe_sibling_pending_keep(r), "a new activation copy would rewrite what the pending job reads");
    r                    = sibling_request();
    r.current_act_serial = 5;
    CHECK(!moe_sibling_pending_keep(r), "the staging was rewritten since the pending job read it");
    r          = sibling_request();
    r.op_first = 0;
    CHECK(!moe_sibling_pending_keep(r), "an overlapping reservation would overwrite the pending output");
    r          = sibling_request();
    r.op_count = 0;
    CHECK(!moe_sibling_pending_keep(r), "an op with no CPU rows has nothing to overlap");
    r                   = sibling_request();
    r.same_row_geometry = false;
    CHECK(!moe_sibling_pending_keep(r), "entry spans of different row sizes cannot be compared");
    r                   = sibling_request();
    r.op_first          = 0;
    r.pending_from_pool = false;
    CHECK(moe_sibling_pending_keep(r), "a pending job with its own buffers shares no pool entries");
    return 0;
}

int main() {
    if (test_direct_request() != 0 || test_direct_stamp() != 0 || test_direct_layout() != 0 || test_pool_ring() != 0 ||
        test_gather_runs() != 0 || test_shared_activation() != 0 || test_sibling_pending() != 0) {
        return 1;
    }
    std::printf("OK: moe decode host path decisions\n");
    return 0;
}
