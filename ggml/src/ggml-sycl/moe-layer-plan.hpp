//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"
#include "mem-handle.hpp"

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

namespace ggml_sycl {

enum class moe_layer_executor_kind : uint8_t {
    NONE            = 0,
    MMVQ_COMPAT     = 1,
    DIRECT_XMX      = 2,
    FUSED_LAYER     = 3,
    MIXED_RESIDENCY = 4,
};

inline const char * moe_layer_executor_kind_name(moe_layer_executor_kind kind) {
    switch (kind) {
        case moe_layer_executor_kind::NONE:
            return "none";
        case moe_layer_executor_kind::MMVQ_COMPAT:
            return "mmvq-compat";
        case moe_layer_executor_kind::DIRECT_XMX:
            return "direct-xmx";
        case moe_layer_executor_kind::FUSED_LAYER:
            return "fused-layer";
        case moe_layer_executor_kind::MIXED_RESIDENCY:
            return "mixed-residency";
    }
    return "unknown";
}

enum class moe_layer_reject_reason : uint8_t {
    NONE = 0,
    TOPOLOGY,
    TYPE,
    SHAPE,
    LAYOUT,
    MISSING_HANDLE,
    HANDLE_RESOLVE,
    RESIDENCY,
    CAPABILITY,
    GRAPH,
    CORRECTNESS_GUARD,
    DIAGNOSTIC_OVERRIDE,
};

inline const char * moe_layer_reject_reason_name(moe_layer_reject_reason reason) {
    switch (reason) {
        case moe_layer_reject_reason::NONE:
            return "none";
        case moe_layer_reject_reason::TOPOLOGY:
            return "topology";
        case moe_layer_reject_reason::TYPE:
            return "type";
        case moe_layer_reject_reason::SHAPE:
            return "shape";
        case moe_layer_reject_reason::LAYOUT:
            return "layout";
        case moe_layer_reject_reason::MISSING_HANDLE:
            return "missing-handle";
        case moe_layer_reject_reason::HANDLE_RESOLVE:
            return "handle-resolve";
        case moe_layer_reject_reason::RESIDENCY:
            return "residency";
        case moe_layer_reject_reason::CAPABILITY:
            return "capability";
        case moe_layer_reject_reason::GRAPH:
            return "graph";
        case moe_layer_reject_reason::CORRECTNESS_GUARD:
            return "correctness-guard";
        case moe_layer_reject_reason::DIAGNOSTIC_OVERRIDE:
            return "diagnostic-override";
    }
    return "unknown";
}

struct moe_gate_up_pair {
    const ggml_tensor * up_weight       = nullptr;
    const ggml_tensor * gate_weight     = nullptr;
    const ggml_tensor * src1            = nullptr;
    const ggml_tensor * ids             = nullptr;
    const ggml_tensor * up_bias         = nullptr;
    const ggml_tensor * gate_bias       = nullptr;
    const ggml_tensor * down_weight     = nullptr;
    ggml_tensor *       up_dst          = nullptr;
    ggml_tensor *       gate_dst        = nullptr;
    ggml_tensor *       up_biased       = nullptr;
    ggml_tensor *       gate_biased     = nullptr;
    ggml_tensor *       glu_dst         = nullptr;
    ggml_tensor *       down_dst        = nullptr;
    int                 up_index        = -1;
    int                 gate_index      = -1;
    int                 up_bias_index   = -1;
    int                 gate_bias_index = -1;
    int                 glu_index       = -1;
    int                 down_index      = -1;
    enum ggml_glu_op    glu_op          = GGML_GLU_OP_SWIGLU;
};

struct moe_layer_grouped_route_row {
    size_t  entry = 0;
    int64_t iid1  = 0;
    int64_t id    = 0;
};

struct moe_layer_decode_role_plan {
    const ggml_tensor *                weight = nullptr;
    std::vector<int32_t>               expert_ids;
    std::vector<mem_handle>            handles;
    std::vector<moe_layer_grouped_route_row> rows;
    std::vector<sycl::event>           ready_events;

    size_t entries() const { return expert_ids.size(); }

    bool has_handle_coverage() const { return handles.size() == expert_ids.size(); }
};

struct moe_layer_transient_ptr_table_slot {
    int32_t     expert_id       = -1;
    mem_handle  handle;
    sycl::event ready_event;
    bool        has_ready_event = false;
};

struct moe_layer_decode_artifact_plan {
    const char *      role   = nullptr;
    const ggml_tensor * tensor = nullptr;
    mem_handle        handle;
    resolved_ptr      resolved;
};

struct moe_layer_decode_plan {
    int                      layer           = -1;
    int                      layer_hash      = -1;
    int64_t                  n_ids           = 0;
    size_t                   n_entries       = 0;
    ggml_layout_mode         layout          = GGML_LAYOUT_AOS;
    bool                     current_is_gate = false;
    const int32_t *          ids_host        = nullptr;
    const int32_t *          ids_device      = nullptr;
    const ggml_tensor *      src1            = nullptr;
    const ggml_tensor *      ids             = nullptr;
    const moe_gate_up_pair * pair            = nullptr;

    moe_layer_decode_artifact_plan activation;
    moe_layer_decode_artifact_plan ids_control;
    moe_layer_decode_artifact_plan gate_bias;
    moe_layer_decode_artifact_plan up_bias;
    moe_layer_decode_artifact_plan gate_output;
    moe_layer_decode_artifact_plan up_output;
    moe_layer_decode_artifact_plan glu_output;
    moe_layer_decode_artifact_plan down_output;

    moe_layer_decode_role_plan current;
    moe_layer_decode_role_plan partner;
    moe_layer_decode_role_plan gate;
    moe_layer_decode_role_plan up;
    moe_layer_decode_role_plan down;

    bool                         down_eligible = false;
    moe_layer_executor_kind      executor      = moe_layer_executor_kind::NONE;
    moe_layer_reject_reason      reject_reason = moe_layer_reject_reason::NONE;

    bool has_gate_up_handles() const {
        return gate.has_handle_coverage() && up.has_handle_coverage();
    }
};

enum class moe_layer_route_residency : uint8_t {
    MISSING = 0,
    DEVICE  = 1,
    HOST    = 2,
};

struct moe_layer_grouped_route_group {
    int32_t                             expert_id    = -1;
    int                                 device       = mem_handle::HOST_DEVICE;
    ggml_layout_mode                    layout       = GGML_LAYOUT_AOS;
    moe_layer_route_residency           residency    = moe_layer_route_residency::MISSING;
    size_t                              handle_index = SIZE_MAX;
    mem_handle                          handle;
    size_t                              handle_hash  = 0;
    std::vector<moe_layer_grouped_route_row> rows;
};

struct moe_layer_grouped_route_view {
    const ggml_tensor *                         weight               = nullptr;
    const char *                                role_name            = "unknown";
    int                                         submit_device        = -1;
    int64_t                                     top_k                = 0;
    int64_t                                     batch                = 0;
    size_t                                      entries              = 0;
    size_t                                      n_groups             = 0;
    size_t                                      device_rows          = 0;
    size_t                                      host_rows            = 0;
    size_t                                      missing_rows         = 0;
    size_t                                      layout_mismatch_rows = 0;
    size_t                                      max_rows_per_group   = 0;
    size_t                                      ready_events         = 0;
    std::vector<moe_layer_grouped_route_group> groups;
};

struct moe_layer_executor_abi {
    const moe_layer_decode_plan * plan          = nullptr;
    int                           submit_device = -1;
    moe_layer_executor_kind       kind          = moe_layer_executor_kind::NONE;
    moe_layer_reject_reason       reject_reason = moe_layer_reject_reason::NONE;
    std::vector<mem_handle>       retained_handles;
    std::vector<sycl::event>      deps;
};

}  // namespace ggml_sycl
