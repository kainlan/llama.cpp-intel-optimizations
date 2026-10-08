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
// selected-row count (it feeds the layout choice). The per-op request check
// below reads only tensor shapes and flags already in hand.
//
// SYCL-free on purpose so tests/test-sycl-moe-decode-hostpath.cpp can run it
// without a device.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_sycl {

struct moe_decode_direct_stamp {
    uint64_t plan_generation           = 0;
    uint64_t expert_storage_generation = 0;
    int64_t  selected_rows             = 0;
    int      layout                    = 0;
    bool     valid                     = false;
    bool     eligible                  = false;
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
// expert of gate and of up) currently holds. Cleared at every graph boundary.
struct moe_shared_act_record {
    uint64_t serial         = 0;  // bumped on every rewrite of the staging
    uint64_t scatter_serial = 0;  // CPU scatter flushes enqueued before the copy
    size_t   view_offset    = 0;
    size_t   bytes          = 0;
    int      device         = -1;
    bool     valid          = false;
};

// An op may skip its own activation copy when the staging already holds its
// src1 row. same_source: src1's storage handle has the recorded identity. The
// scatter serial matters because the copy's completion is what orders every
// earlier scatter H2D ahead of this op's writes to its pool entries; a scatter
// enqueued after the copy is not covered by it.
inline bool moe_shared_act_reusable(const moe_shared_act_record & r,
                                    bool                          same_source,
                                    uint64_t                      scatter_serial,
                                    size_t                        view_offset,
                                    size_t                        bytes,
                                    int                           device) {
    return r.valid && same_source && r.scatter_serial == scatter_serial && r.view_offset == view_offset &&
           r.bytes == bytes && r.device == device;
}

// May this op's CPU job be issued while the pending job (gate, typically) is
// still running, instead of joining and scattering it first? Each condition
// guards a resource the two jobs would otherwise share.
struct moe_sibling_pending_request {
    bool     pending_active     = false;  // a CPU job is pending in the primary slot
    bool     sibling_slot_free  = false;  // the second slot can hold it
    bool     reuses_activation  = false;  // this op does no activation copy
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

}  // namespace ggml_sycl
