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
    const ggml_tensor * down_bias       = nullptr;
    const ggml_tensor * down_weight     = nullptr;
    const ggml_tensor * moe_weights     = nullptr;
    ggml_tensor *       up_dst          = nullptr;
    ggml_tensor *       gate_dst        = nullptr;
    ggml_tensor *       up_biased       = nullptr;
    ggml_tensor *       gate_biased     = nullptr;
    ggml_tensor *       glu_dst         = nullptr;
    ggml_tensor *       down_dst        = nullptr;
    ggml_tensor *       down_biased     = nullptr;
    ggml_tensor *       weighted_dst    = nullptr;
    int                 up_index        = -1;
    int                 gate_index      = -1;
    int                 up_bias_index   = -1;
    int                 gate_bias_index = -1;
    int                 glu_index       = -1;
    int                 down_index      = -1;
    int                 down_bias_index = -1;
    int                 weighted_index  = -1;
    int                 down_sum_final_index = -1;
    enum ggml_glu_op    glu_op          = GGML_GLU_OP_SWIGLU;
};

struct moe_layer_grouped_route_row {
    size_t  entry = 0;
    int64_t iid1  = 0;
    int64_t id    = 0;
};

struct moe_layer_decode_role_plan {
    const ggml_tensor *                weight = nullptr;
    ggml_layout_mode                   layout = GGML_LAYOUT_AOS;
    std::vector<int32_t>               expert_ids;
    std::vector<mem_handle>            handles;
    std::vector<moe_layer_grouped_route_row> rows;
    std::vector<sycl::event>           ready_events;
    bool                               full_table_static_handles = false;

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

struct moe_layer_persistent_tensor_descriptor {
    const char *        role   = nullptr;
    const ggml_tensor * tensor = nullptr;
    mem_handle          handle;
};

struct moe_layer_persistent_role_descriptor {
    const char *           role   = nullptr;
    const ggml_tensor *    weight = nullptr;
    ggml_layout_mode       layout = GGML_LAYOUT_AOS;
    std::vector<mem_handle>  expert_handles;
    std::vector<sycl::event> ready_events;

    size_t experts() const { return expert_handles.size(); }
};

struct moe_layer_persistent_descriptor {
    int                                    layer         = -1;
    int                                    submit_device = -1;
    int64_t                                top_k         = 0;
    moe_layer_persistent_tensor_descriptor activation;
    moe_layer_persistent_tensor_descriptor ids_control;
    moe_layer_persistent_role_descriptor   gate;
    moe_layer_persistent_role_descriptor   up;
    moe_layer_persistent_role_descriptor   down;

    size_t retained_handles() const {
        size_t n = 0;
        n += activation.handle.valid() ? 1 : 0;
        n += ids_control.handle.valid() ? 1 : 0;
        n += gate.expert_handles.size();
        n += up.expert_handles.size();
        n += down.expert_handles.size();
        return n;
    }

    bool complete() const {
        return layer >= 0 && submit_device >= 0 && activation.handle.valid() && ids_control.handle.valid() &&
               !gate.expert_handles.empty() && !up.expert_handles.empty() && !down.expert_handles.empty();
    }
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

enum class moe_route_phase : uint8_t {
    DECODE = 0,
    PROMPT = 1,
};

inline const char * moe_route_phase_name(moe_route_phase phase) {
    switch (phase) {
        case moe_route_phase::DECODE:
            return "decode";
        case moe_route_phase::PROMPT:
            return "prompt";
    }
    return "unknown";
}

enum class moe_route_kernel : uint8_t {
    NONE          = 0,
    HOST_CPU      = 1,
    MMVQ_COMPAT   = 2,
    SECONDARY_SOA = 3,
    XMX_TILED     = 4,
    MXFP4_I8      = 5,
    MXFP4_DPAS    = 6,
};

inline const char * moe_route_kernel_name(moe_route_kernel kernel) {
    switch (kernel) {
        case moe_route_kernel::NONE:
            return "none";
        case moe_route_kernel::HOST_CPU:
            return "host-cpu";
        case moe_route_kernel::MMVQ_COMPAT:
            return "mmvq-compat";
        case moe_route_kernel::SECONDARY_SOA:
            return "secondary-soa";
        case moe_route_kernel::XMX_TILED:
            return "xmx-tiled";
        case moe_route_kernel::MXFP4_I8:
            return "mxfp4-i8";
        case moe_route_kernel::MXFP4_DPAS:
            return "mxfp4-dpas";
    }
    return "unknown";
}

struct moe_route_capability {
    bool                    supported             = false;
    bool                    local_device          = false;
    bool                    requires_host_staging = false;
    moe_route_phase         phase                 = moe_route_phase::DECODE;
    moe_route_kernel        kernel                = moe_route_kernel::NONE;
    moe_layer_reject_reason reject_reason         = moe_layer_reject_reason::NONE;
    const char *            reason                = "uninitialized";
};

struct moe_layer_grouped_route_group {
    int32_t                             expert_id    = -1;
    int                                 device       = mem_handle::HOST_DEVICE;
    ggml_layout_mode                    layout       = GGML_LAYOUT_AOS;
    moe_layer_route_residency           residency    = moe_layer_route_residency::MISSING;
    moe_route_phase                     phase        = moe_route_phase::DECODE;
    size_t                              handle_index = SIZE_MAX;
    mem_handle                          handle;
    size_t                              handle_hash  = 0;
    std::vector<sycl::event>            deps;
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
    std::vector<sycl::event>                    deps;
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
