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

#include <cstdint>

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

}  // namespace ggml_sycl
