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

// Stand-ins for graph nodes: only their identity matters.
static const int k_gate_dst  = 0;
static const int k_up_dst    = 0;
static const int k_src1      = 0;
static const int k_next_src1 = 0;
static const int k_down_dst  = 0;

static int test_direct_retry_cap() {
    using ggml_sycl::moe_decode_direct_stamp_current;
    using ggml_sycl::moe_decode_direct_stamp_settle;
    const uint32_t limit = ggml_sycl::moe_decode_direct_retry_limit;
    const auto     retry = ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_RETRY;

    ggml_sycl::moe_decode_direct_stamp s{};
    for (uint32_t i = 1; i < limit; ++i) {
        CHECK(moe_decode_direct_stamp_settle(s, 3, 7, 10, 0, retry) == retry, "a retry below the cap stays a retry");
        CHECK(!moe_decode_direct_stamp_current(s, 3, 7, 10), "a retry below the cap is not remembered");
    }
    CHECK(moe_decode_direct_stamp_settle(s, 3, 7, 10, 0, retry) == ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED,
          "the retry that reaches the cap settles as a refusal");
    CHECK(moe_decode_direct_stamp_current(s, 3, 7, 10) && !s.eligible, "the capped refusal is remembered");

    // A generation bump starts a new count: the new storage gets its own tries.
    ggml_sycl::moe_decode_direct_stamp t{};
    for (uint32_t i = 1; i < limit; ++i) {
        moe_decode_direct_stamp_settle(t, 3, 7, 10, 0, retry);
    }
    for (uint32_t i = 1; i < limit; ++i) {
        CHECK(moe_decode_direct_stamp_settle(t, 3, 8, 10, 0, retry) == retry,
              "a storage generation bump resets the retry count");
    }
    CHECK(moe_decode_direct_stamp_settle(t, 3, 8, 10, 0, retry) == ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_REFUSED,
          "the reset count reaches the cap again for the new generation");

    // A decision in between also starts a new count.
    ggml_sycl::moe_decode_direct_stamp u{};
    for (uint32_t i = 1; i < limit; ++i) {
        moe_decode_direct_stamp_settle(u, 3, 7, 10, 0, retry);
    }
    moe_decode_direct_stamp_settle(u, 3, 7, 10, 0, ggml_sycl::MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE);
    CHECK(moe_decode_direct_stamp_settle(u, 3, 7, 10, 0, retry) == retry,
          "an eligible decision resets the retry count");
    return 0;
}

static int test_shared_act_sibling() {
    using ggml_sycl::moe_shared_act_sibling_of;
    ggml_sycl::moe_gate_up_nodes pair;
    pair.gate_dst  = &k_gate_dst;
    pair.gate_src1 = &k_src1;
    pair.up_dst    = &k_up_dst;
    pair.up_src1   = &k_src1;
    CHECK(moe_shared_act_sibling_of(&pair, &k_gate_dst, &k_src1) == &k_up_dst, "gate's sibling is up");
    CHECK(moe_shared_act_sibling_of(&pair, &k_up_dst, &k_src1) == &k_gate_dst, "up's sibling is gate");

    ggml_sycl::moe_gate_up_nodes split = pair;
    split.up_src1                      = &k_next_src1;
    CHECK(moe_shared_act_sibling_of(&split, &k_gate_dst, &k_src1) == nullptr,
          "gate and up reading different src1 nodes are not siblings");
    CHECK(moe_shared_act_sibling_of(&pair, &k_gate_dst, &k_next_src1) == nullptr,
          "an op reading another src1 than the scanned pair has no sibling");

    ggml_sycl::moe_gate_up_nodes fused;  // one gate_up node: the scan files it as gate
    fused.gate_dst  = &k_gate_dst;
    fused.gate_src1 = &k_src1;
    CHECK(moe_shared_act_sibling_of(&fused, &k_gate_dst, &k_src1) == nullptr, "a fused gate_up op has no sibling");

    CHECK(moe_shared_act_sibling_of(&pair, &k_down_dst, &k_src1) == nullptr, "down has no sibling");
    CHECK(moe_shared_act_sibling_of(&pair, &k_next_src1, &k_src1) == nullptr,
          "an op that is not a scanned node has no sibling");
    CHECK(moe_shared_act_sibling_of(nullptr, &k_gate_dst, &k_src1) == nullptr,
          "a layer missing from the scan has no sibling");
    CHECK(moe_shared_act_sibling_of(&pair, nullptr, &k_src1) == nullptr, "no op, no sibling");
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

// The residency callback for the GLU walk tests: ctx is the one weight that executes on the host.
static bool weight_is(const ggml_tensor * w, void * ctx) {
    return w == static_cast<const ggml_tensor *>(ctx);
}

// GLU placement: only a host MUL_MAT producer is host-produced; a MUL_MAT_ID is device-produced
// and ends the walk (llama.cpp-z4kd).  Plain tensors, no context: the walk reads op and src only.
static int test_glu_input_host_produced() {
    using ggml_sycl::moe_glu_input_host_produced;
    ggml_tensor host_w{};
    ggml_tensor dev_w{};
    ggml_tensor act{};
    act.op         = GGML_OP_NONE;
    void * on_host = &host_w;

    // A dense FFN whose gate/up weights are on the host: GLU follows its producers.
    ggml_tensor gate{};
    gate.op     = GGML_OP_MUL_MAT;
    gate.src[0] = &host_w;
    gate.src[1] = &act;
    ggml_tensor glu{};
    glu.op     = GGML_OP_GLU;
    glu.src[0] = &gate;
    CHECK(moe_glu_input_host_produced(&glu, weight_is, on_host), "a GLU fed by a host MUL_MAT is host-produced");

    // The same with device weights.
    gate.src[0] = &dev_w;
    CHECK(!moe_glu_input_host_produced(&glu, weight_is, on_host), "a GLU fed by a device MUL_MAT is not host-produced");

    // A MoE layer whose experts are on the host: the MUL_MAT_ID writes device memory.
    ggml_tensor mmid{};
    mmid.op     = GGML_OP_MUL_MAT_ID;
    mmid.src[0] = &host_w;
    mmid.src[1] = &act;
    glu.src[0]  = &mmid;
    CHECK(!moe_glu_input_host_produced(&glu, weight_is, on_host),
          "a GLU fed by a host-expert MUL_MAT_ID is not host-produced");

    // A host MUL_MAT upstream of the MUL_MAT_ID's activation is behind it, so it does not count.
    ggml_tensor up_proj{};
    up_proj.op     = GGML_OP_MUL_MAT;
    up_proj.src[0] = &host_w;
    up_proj.src[1] = &act;
    mmid.src[0]    = &dev_w;
    mmid.src[1]    = &up_proj;
    CHECK(!moe_glu_input_host_produced(&glu, weight_is, on_host), "the walk does not descend through a MUL_MAT_ID");

    // Through an elementwise op, a host MUL_MAT still counts (the dense case keeps its reach).
    ggml_tensor scale{};
    scale.op     = GGML_OP_MUL;
    scale.src[0] = &up_proj;
    scale.src[1] = &act;
    glu.src[0]   = &scale;
    CHECK(moe_glu_input_host_produced(&glu, weight_is, on_host), "a host MUL_MAT behind an elementwise op counts");

    // GPT-OSS shape: the GLU reads an expert bias add (ADD_ID) over a host-expert MUL_MAT_ID.
    mmid.src[0] = &host_w;
    mmid.src[1] = &act;
    ggml_tensor bias_add{};
    bias_add.op     = GGML_OP_ADD_ID;
    bias_add.src[0] = &mmid;
    bias_add.src[1] = &act;
    glu.src[0]      = &bias_add;
    CHECK(!moe_glu_input_host_produced(&glu, weight_is, on_host),
          "a GLU fed by a bias add over a host-expert MUL_MAT_ID is not host-produced");

    // A walk that would need more than MOE_GLU_INPUT_WALK_MAX_VISITED nodes gives up as device-produced:
    // three levels of ten-way fan-out (1 + 10 + 100 + 1000 nodes), the host MUL_MAT reachable only last.
    static ggml_tensor fan1[10];
    static ggml_tensor fan2[100];
    static ggml_tensor fan3[1000];
    ggml_tensor        wide{};
    wide.op = GGML_OP_GLU;
    for (int i = 0; i < 10; ++i) {
        fan1[i]     = ggml_tensor{};
        fan1[i].op  = GGML_OP_ADD;
        wide.src[i] = &fan1[i];
        for (int j = 0; j < 10; ++j) {
            ggml_tensor & m = fan2[i * 10 + j];
            m               = ggml_tensor{};
            m.op            = GGML_OP_ADD;
            fan1[i].src[j]  = &m;
            for (int k = 0; k < 10; ++k) {
                ggml_tensor & l = fan3[(i * 10 + j) * 10 + k];
                l               = ggml_tensor{};
                l.op            = GGML_OP_NONE;
                m.src[k]        = &l;
            }
        }
    }
    ggml_tensor & last = fan3[999];
    last.op            = GGML_OP_MUL_MAT;
    last.src[0]        = &host_w;
    last.src[1]        = &act;
    CHECK(!moe_glu_input_host_produced(&wide, weight_is, on_host),
          "a walk past the visited capacity ends as device-produced");
    fan3[0].op     = GGML_OP_MUL_MAT;
    fan3[0].src[0] = &host_w;
    fan3[0].src[1] = &act;
    CHECK(moe_glu_input_host_produced(&wide, weight_is, on_host), "a host MUL_MAT within the capacity still counts");
    return 0;
}

// The submitting-thread wait census (llama.cpp-z4kd): which class a wait is counted under.
static int test_wait_census_classes() {
    using namespace ggml_sycl;
    CHECK(moe_hostpath_join_class("ffn_moe_down-29") == MOE_WAIT_B5, "a down job join is B5");
    CHECK(moe_hostpath_join_class("ffn_moe_gate-29") == MOE_WAIT_B3, "a gate job join is B3");
    CHECK(moe_hostpath_join_class("ffn_moe_up-29") == MOE_WAIT_B3, "an up job join is B3");
    CHECK(moe_hostpath_join_class(nullptr) == MOE_WAIT_B3, "an unnamed job join is B3");

    const int none = -1;
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B1, nullptr, none) == MOE_WAIT_B1, "a readback is B1");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B6, nullptr, none) == MOE_WAIT_B6, "a scatter wait is B6");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_JOIN, "ffn_moe_down-3", none) == MOE_WAIT_B5,
          "a join is classified by the joined op's name");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_JOIN, "ffn_moe_gate-3", none) == MOE_WAIT_B3,
          "a gate join outside any flush context is B3");

    // Inside the graph-boundary flush every wait is B7.
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B1, nullptr, MOE_WAIT_B7) == MOE_WAIT_B7, "boundary: a readback is B7");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B6, nullptr, MOE_WAIT_B7) == MOE_WAIT_B7, "boundary: a scatter is B7");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_JOIN, "ffn_moe_down-3", MOE_WAIT_B7) == MOE_WAIT_B7,
          "boundary: a join is B7");

    // Inside the hot-group flush a job join is B4b; other waits keep their class.
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_JOIN, "ffn_moe_down-3", MOE_WAIT_B4B) == MOE_WAIT_B4B,
          "hot group: a down join is B4b");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_JOIN, "ffn_moe_gate-3", MOE_WAIT_B4B) == MOE_WAIT_B4B,
          "hot group: a gate/up join (prefill) is B4b");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B6, nullptr, MOE_WAIT_B4B) == MOE_WAIT_B6,
          "hot group: a scatter wait stays B6");
    CHECK(moe_hostpath_wait_classify(MOE_WAIT_B2, nullptr, MOE_WAIT_B4B) == MOE_WAIT_B2,
          "hot group: an activation wait stays B2");
    return 0;
}

int main() {
    if (test_direct_request() != 0 || test_direct_stamp() != 0 || test_direct_layout() != 0 ||
        test_direct_retry_cap() != 0 || test_pool_ring() != 0 || test_gather_runs() != 0 ||
        test_shared_activation() != 0 || test_shared_act_sibling() != 0 || test_sibling_pending() != 0 ||
        test_glu_input_host_produced() != 0 || test_wait_census_classes() != 0) {
        return 1;
    }
    std::printf("OK: moe decode host path decisions\n");
    return 0;
}
