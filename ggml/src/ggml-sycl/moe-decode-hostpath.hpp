#pragma once

// Host-side decisions of the batch-1 (decode) MUL_MAT_ID path (llama.cpp-yx28).
//
// A decode op whose experts are all resident on the executing device can run
// from a device-resident full-coverage expert pointer table and the device ids
// tensor, with no ids readback, no per-op table upload and no host batch-ids
// upload. Deciding that eligibility walks the placement plan and every
// expert's storage, so the decision is cached per tensor and per device. The
// stamp holds what can change it: the replan epoch (a MID_LOAD_REPLAN bumps
// it), the expert storage generation (a storage rewrite bumps it) and the
// selected-row count the decision was made for. The per-op request check
// below reads only tensor shapes and flags already in hand.
//
// SYCL-free on purpose so tests/test-sycl-moe-decode-hostpath.cpp can run it
// without a device.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ggml_sycl {

struct moe_decode_direct_stamp {
    uint64_t plan_generation           = 0;
    uint64_t expert_storage_generation = 0;
    int64_t  selected_rows             = 0;
    int      layout                    = 0;
    bool     valid                     = false;
    bool     eligible                  = false;
    uint32_t retries                   = 0;  // RETRY outcomes in a row for these inputs
};

inline bool moe_decode_direct_stamp_current(const moe_decode_direct_stamp & s,
                                            uint64_t                        plan_generation,
                                            uint64_t                        expert_storage_generation,
                                            int64_t                         selected_rows) {
    return s.valid && s.plan_generation == plan_generation &&
           s.expert_storage_generation == expert_storage_generation && s.selected_rows == selected_rows;
}

inline void moe_decode_direct_stamp_record(moe_decode_direct_stamp & s,
                                           uint64_t                  plan_generation,
                                           uint64_t                  expert_storage_generation,
                                           int64_t                   selected_rows,
                                           int                       layout,
                                           bool                      eligible) {
    s.plan_generation           = plan_generation;
    s.expert_storage_generation = expert_storage_generation;
    s.selected_rows             = selected_rows;
    s.layout                    = layout;
    s.valid                     = true;
    s.eligible                  = eligible;
}

// How one eligibility decision ended. A refusal follows from the tensor, the
// device or the materialized storage, so it holds until the stamp's inputs
// change. A retry is a failure of the attempt itself (the table upload could
// not allocate), so the next op decides again, up to
// moe_decode_direct_retry_limit times in a row for the same inputs. The retry
// that reaches the limit settles as a refusal, so a failure that never clears
// stops costing a full decision per op. New inputs (a replan or a storage
// rewrite) start a new count.
enum moe_decode_direct_outcome {
    MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE,
    MOE_DECODE_DIRECT_OUTCOME_REFUSED,
    MOE_DECODE_DIRECT_OUTCOME_RETRY,
};

constexpr uint32_t moe_decode_direct_retry_limit = 8;

inline moe_decode_direct_outcome moe_decode_direct_stamp_settle(moe_decode_direct_stamp & s,
                                                                uint64_t                  plan_generation,
                                                                uint64_t                  expert_storage_generation,
                                                                int64_t                   selected_rows,
                                                                int                       layout,
                                                                moe_decode_direct_outcome outcome) {
    if (outcome == MOE_DECODE_DIRECT_OUTCOME_RETRY) {
        const bool same_inputs = s.plan_generation == plan_generation &&
                                 s.expert_storage_generation == expert_storage_generation &&
                                 s.selected_rows == selected_rows;
        const uint32_t retries = same_inputs ? s.retries + 1 : 1;
        if (retries >= moe_decode_direct_retry_limit) {
            moe_decode_direct_stamp_record(s, plan_generation, expert_storage_generation, selected_rows, layout,
                                           /*eligible=*/false);
            s.retries = retries;
            return MOE_DECODE_DIRECT_OUTCOME_REFUSED;
        }
        s.plan_generation           = plan_generation;
        s.expert_storage_generation = expert_storage_generation;
        s.selected_rows             = selected_rows;
        s.valid                     = false;
        s.retries                   = retries;
        return outcome;
    }
    moe_decode_direct_stamp_record(s, plan_generation, expert_storage_generation, selected_rows, layout,
                                   outcome == MOE_DECODE_DIRECT_OUTCOME_ELIGIBLE);
    s.retries = 0;
    return outcome;
}

// The route's layout is the one expert 0 is materialized in on the device:
// what the unified cache loaded, not what a policy would pick. `layouts` lists
// the layouts of expert 0's records on that device that the batched kernel
// reads; a copy held for another executor (a prompt-processing alternate, say)
// is not in it. Exactly one distinct layout is the answer; none, or more than
// one, leaves the route without one.
inline bool moe_decode_direct_layout_from_materialized(const std::vector<int> & layouts, int * layout) {
    if (layouts.empty() || !layout) {
        return false;
    }
    for (int candidate : layouts) {
        if (candidate != layouts.front()) {
            return false;
        }
    }
    *layout = layouts.front();
    return true;
}

struct moe_decode_direct_request {
    int64_t src1_tokens            = 0;      // src1->ne[2]
    int64_t ids_tokens             = 0;      // ids->ne[1]
    int64_t ids_selected           = 0;      // ids->ne[0]
    bool    graph_recording        = false;
    bool    layout_override        = false;  // diagnostic GGML_SYCL layout override active
    bool    dedicated_decode_route = false;  // MXFP4 / Q1_0 / NVFP4 own their decode executors
};

inline bool moe_decode_direct_request_admissible(const moe_decode_direct_request & r) {
    return r.src1_tokens == 1 && r.ids_tokens == 1 && r.ids_selected > 0 && !r.graph_recording && !r.layout_override &&
           !r.dedicated_decode_route;
}

// ---------------------------------------------------------------------------
// CPU-expert decode ops (llama.cpp-yx28 stage 2): host-tiered experts run on
// the CPU from PinnedBufferPool staging, and the results scatter back H2D.
// ---------------------------------------------------------------------------

// The PinnedBufferPool ring: the first entry reserve(n) hands out from
// `cursor` in a pool of `capacity` entries. PinnedBufferPool::reserve() and
// the sibling decision below, which predicts a reservation, share it.
inline size_t moe_pool_reserve_first(size_t cursor, size_t n, size_t capacity) {
    return cursor + n > capacity ? 0 : cursor;
}

inline bool moe_pool_spans_disjoint(size_t a_first, size_t a_count, size_t b_first, size_t b_count) {
    return a_first + a_count <= b_first || b_first + b_count <= a_first;
}

// One copy per run of rows whose sources are contiguous. Destination row i
// lands at dst_base + i * row_bytes, so destinations are always contiguous.
struct moe_gather_run {
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t bytes      = 0;
};

inline void moe_gather_runs_build(const std::vector<size_t> &   src_offsets,
                                  size_t                        dst_base,
                                  size_t                        row_bytes,
                                  std::vector<moe_gather_run> & runs) {
    runs.clear();
    for (size_t i = 0; i < src_offsets.size(); ++i) {
        const size_t dst = dst_base + i * row_bytes;
        if (!runs.empty()) {
            moe_gather_run & last = runs.back();
            if (last.src_offset + last.bytes == src_offsets[i] && last.dst_offset + last.bytes == dst) {
                last.bytes += row_bytes;
                continue;
            }
        }
        moe_gather_run run;
        run.src_offset = src_offsets[i];
        run.dst_offset = dst;
        run.bytes      = row_bytes;
        runs.push_back(run);
    }
}

// What the shared activation staging (one src1 row, read by every host
// expert of gate and of up) currently holds.
struct moe_shared_act_record {
    const void * src1_tensor    = nullptr;  // graph node the row was copied from
    const void * sibling_dst    = nullptr;  // the one op that may reuse it
    uint64_t     graph_epoch    = 0;
    uint64_t     serial         = 0;        // never repeats: one value per rewrite of the staging
    uint64_t     scatter_serial = 0;        // CPU scatter flushes enqueued before the copy
    size_t       view_offset    = 0;
    size_t       bytes          = 0;
    int          device         = -1;
    bool         valid          = false;
};

// A layer's gate and up MUL_MAT_ID nodes as the graph scan found them, with
// the src1 node each reads. Identities only; nothing is dereferenced.
struct moe_gate_up_nodes {
    const void * gate_dst  = nullptr;
    const void * gate_src1 = nullptr;
    const void * up_dst    = nullptr;
    const void * up_src1   = nullptr;
};

// The one op that may reuse the activation row `dst` copies to host: the
// other half of its layer's gate/up pair, and only when gate, up and `dst`
// all read the same src1 node. Anything else -- down, a fused gate_up (the
// scan finds no up), an op the scan did not find, a layer it has no pair for
// (`pair` null) -- has no sibling and makes its own copy.
inline const void * moe_shared_act_sibling_of(const moe_gate_up_nodes * pair, const void * dst, const void * dst_src1) {
    if (!pair || !dst || !dst_src1 || !pair->gate_dst || !pair->up_dst || pair->gate_src1 != dst_src1 ||
        pair->up_src1 != dst_src1) {
        return nullptr;
    }
    if (dst == pair->gate_dst) {
        return pair->up_dst;
    }
    return dst == pair->up_dst ? pair->gate_dst : nullptr;
}

struct moe_shared_act_query {
    const void * src1_tensor    = nullptr;
    const void * op_dst         = nullptr;
    bool         same_source    = false;  // src1's storage handle has the recorded identity
    uint64_t     graph_epoch    = 0;
    uint64_t     scatter_serial = 0;
    size_t       view_offset    = 0;
    size_t       bytes          = 0;
    int          device         = -1;
};

// An op may skip its own activation copy when the staging already holds its
// src1 row. The storage location alone does not say that: the compute buffer
// reuses offsets, so the next layer's input can sit where this one's did. The
// row is this op's only if it was copied from the same src1 node, for this
// op's gate/up sibling, within the same graph compute. The scatter serial
// matters because the copy's completion is what orders every earlier scatter
// H2D ahead of this op's writes to its pool entries; a scatter enqueued after
// the copy is not covered by it.
inline bool moe_shared_act_reusable(const moe_shared_act_record & r, const moe_shared_act_query & q) {
    return r.valid && r.src1_tensor != nullptr && r.src1_tensor == q.src1_tensor && r.sibling_dst != nullptr &&
           r.sibling_dst == q.op_dst && r.graph_epoch == q.graph_epoch && q.same_source &&
           r.scatter_serial == q.scatter_serial && r.view_offset == q.view_offset && r.bytes == q.bytes &&
           r.device == q.device;
}

// May this op's CPU job be issued while the pending job (gate, typically) is
// still running, instead of joining and scattering it first? Each condition
// guards a resource the two jobs would otherwise share.
struct moe_sibling_pending_request {
    bool     pending_active     = false;  // a CPU job is pending in the primary slot
    bool     sibling_slot_free  = false;  // the second slot can hold it
    bool     reuses_activation  = false;  // this op reads the pending job's activation copy, making none
    uint64_t pending_act_serial = 0;      // staging contents the pending job reads
    uint64_t current_act_serial = 0;      // staging contents now
    bool     pending_from_pool  = false;
    bool     op_from_pool       = false;
    size_t   pending_first      = 0;      // pool entries the pending job writes and its scatter reads
    size_t   pending_count      = 0;
    size_t   op_first           = 0;      // this op's predicted reservation
    size_t   op_count           = 0;
    bool     same_row_geometry  = false;  // equal K and N, so entry spans compare
};

inline bool moe_sibling_pending_keep(const moe_sibling_pending_request & r) {
    if (!r.pending_active || !r.sibling_slot_free || !r.reuses_activation || r.op_count == 0 ||
        r.pending_act_serial != r.current_act_serial) {
        return false;
    }
    if (!r.pending_from_pool || !r.op_from_pool) {
        return true;  // one side owns separate buffers
    }
    return r.same_row_geometry && moe_pool_spans_disjoint(r.pending_first, r.pending_count, r.op_first, r.op_count);
}

// Where a GLU runs follows where its input was produced.  The input is
// host-produced when, within 8 levels, it comes from a MUL_MAT whose weight
// executes on the host: that MUL_MAT runs on the CPU backend, so the GLU stays
// there with it.  A MUL_MAT_ID ends the walk and counts as device-produced, host
// experts or not: this backend always admits it and writes its output to device
// memory (the host experts' rows arrive by the scatter H2D).  Counting it as a
// host producer moved the GLU of every host-expert layer to the CPU backend,
// which costs two scheduler copies and two split boundaries per layer.  The
// split also starts the down projection in a new graph_compute, whose entry
// clears the per-layer ids cache, so down read the expert ids back a second
// time (llama.cpp-z4kd).
//
// The residency question is a callback (function pointer plus context) so the
// walk runs without a device.  The walk remembers up to
// MOE_GLU_INPUT_WALK_MAX_VISITED nodes; a graph that needs more ends the walk as
// device-produced, which keeps the GLU on SYCL and costs at most a copy.
typedef bool (*moe_weight_on_host_fn)(const ggml_tensor * weight, void * ctx);

constexpr int MOE_GLU_INPUT_WALK_MAX_VISITED = 256;

struct moe_glu_input_walk_state {
    moe_weight_on_host_fn weight_on_host;
    void *                ctx;
    const ggml_tensor *   visited[MOE_GLU_INPUT_WALK_MAX_VISITED];
    int                   n_visited;
};

inline bool moe_glu_input_walk(const ggml_tensor * t, moe_glu_input_walk_state & st, int depth) {
    if (!t || depth > 8) {
        return false;
    }
    for (int i = 0; i < st.n_visited; ++i) {
        if (st.visited[i] == t) {
            return false;
        }
    }
    if (st.n_visited == MOE_GLU_INPUT_WALK_MAX_VISITED) {
        return false;
    }
    st.visited[st.n_visited++] = t;
    if (t->op == GGML_OP_MUL_MAT_ID) {
        return false;
    }
    if (t->op == GGML_OP_MUL_MAT && t->src[0] != nullptr && st.weight_on_host(t->src[0], st.ctx)) {
        return true;
    }
    for (int s = 0; s < GGML_MAX_SRC && t->src[s] != nullptr; ++s) {
        if (moe_glu_input_walk(t->src[s], st, depth + 1)) {
            return true;
        }
    }
    return false;
}

inline bool moe_glu_input_host_produced(const ggml_tensor * op, moe_weight_on_host_fn weight_on_host, void * ctx) {
    moe_glu_input_walk_state st;
    st.weight_on_host = weight_on_host;
    st.ctx            = ctx;
    st.n_visited      = 0;
    return moe_glu_input_walk(op, st, 0);
}

// Wait classes of the host-expert MoE decode path's submitting-thread census
// (llama.cpp-z4kd), printed under GGML_SYCL_MOE_IDS_COPY_TRACE:
//   B1  expert-id readback (synchronous D2H)
//   B2  activation copy and host-lease producer waits before a CPU job
//   B3  join of a gate/up CPU job at a flush
//   B4  down-projection activation gather
//   B4b join of the hot down group right after issuing it
//   B5  join of a down CPU job at a flush
//   B6  wait for an earlier scatter H2D (flush prologue, sibling join)
//   B7  any of the above inside the graph-boundary flush or drain
// MOE_WAIT_JOIN is not a class: it marks a CPU job join whose class (B3 or B5)
// is read from the joined op's name, and only when the census is on.
enum moe_hostpath_wait_class : int {
    MOE_WAIT_JOIN = -1,
    MOE_WAIT_B1,
    MOE_WAIT_B2,
    MOE_WAIT_B3,
    MOE_WAIT_B4,
    MOE_WAIT_B4B,
    MOE_WAIT_B5,
    MOE_WAIT_B6,
    MOE_WAIT_B7,
    MOE_WAIT_COUNT,
};

// A CPU job join is B5 for a down projection and B3 otherwise.
inline int moe_hostpath_join_class(const char * joined_name) {
    return joined_name && strstr(joined_name, "down") ? MOE_WAIT_B5 : MOE_WAIT_B3;
}

// The class one wait is counted under.  `context` is the class an enclosing
// flush forces (-1 for none): inside the graph-boundary flush every wait is B7;
// inside the hot-group flush a job join (B3 or B5) is B4b and other waits keep
// their own class.
inline int moe_hostpath_wait_classify(int natural, const char * joined_name, int context) {
    const int cls = natural == MOE_WAIT_JOIN ? moe_hostpath_join_class(joined_name) : natural;
    if (context == MOE_WAIT_B7) {
        return MOE_WAIT_B7;
    }
    if (context == MOE_WAIT_B4B && (cls == MOE_WAIT_B3 || cls == MOE_WAIT_B5)) {
        return MOE_WAIT_B4B;
    }
    return cls;
}

}  // namespace ggml_sycl
