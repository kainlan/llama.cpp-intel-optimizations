//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"
#include "mem-handle.hpp"
#include "residency-plan.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
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

enum class moe_descriptor_reject_reason : uint8_t {
    NONE = 0,
    INVALID_HANDLE,
    IDENTITY_MISMATCH,
    GENERATION_MISMATCH,
    LAYOUT_MISMATCH,
    DEVICE_MISMATCH,
};

inline const char * moe_descriptor_reject_reason_name(moe_descriptor_reject_reason reason) {
    switch (reason) {
        case moe_descriptor_reject_reason::NONE:
            return "none";
        case moe_descriptor_reject_reason::INVALID_HANDLE:
            return "invalid-handle";
        case moe_descriptor_reject_reason::IDENTITY_MISMATCH:
            return "identity-mismatch";
        case moe_descriptor_reject_reason::GENERATION_MISMATCH:
            return "generation-mismatch";
        case moe_descriptor_reject_reason::LAYOUT_MISMATCH:
            return "layout-mismatch";
        case moe_descriptor_reject_reason::DEVICE_MISMATCH:
            return "device-mismatch";
    }
    return "unknown";
}

struct moe_residency_descriptor_entry {
    std::string      name;
    mem_handle       handle;
    ggml_layout_mode expected_layout      = GGML_LAYOUT_AOS;
    ggml_layout_mode recorded_layout      = GGML_LAYOUT_AOS;
    int              expected_device      = mem_handle::HOST_DEVICE;
    size_t           stable_identity_hash = 0;
    uint64_t         generation           = 0;
};

class moe_residency_descriptor {
  public:
    void add_entry(const char * name, const mem_handle & handle, ggml_layout_mode layout, int device) {
        moe_residency_descriptor_entry entry;
        entry.name                 = name ? name : "";
        entry.handle               = handle;
        entry.expected_layout      = layout;
        entry.recorded_layout      = layout;
        entry.expected_device      = device;
        entry.stable_identity_hash = handle.stable_identity_hash();
        entry.generation           = handle.generation();
        entries_.push_back(entry);
    }

    void add_entry_for_test(const char * name, const mem_handle & handle, ggml_layout_mode layout, int device) {
        add_entry(name, handle, layout, device);
    }

    size_t retained_handle_count() const { return entries_.size(); }

    const std::vector<moe_residency_descriptor_entry> & entries() const { return entries_; }

    moe_descriptor_reject_reason last_reject_reason() const { return last_reject_; }

    bool validate_for_replay() const {
        last_reject_ = moe_descriptor_reject_reason::NONE;
        for (const moe_residency_descriptor_entry & entry : entries_) {
            if (!entry.handle.has_stable_owner_identity()) {
                last_reject_ = moe_descriptor_reject_reason::INVALID_HANDLE;
                residency_diagnostics_record_stale_descriptor_invalid_handle_for_test();
                return false;
            }
            if (entry.handle.is_arena() && entry.handle.generation() != entry.generation) {
                last_reject_ = moe_descriptor_reject_reason::GENERATION_MISMATCH;
                residency_diagnostics_record_stale_descriptor_generation_mismatch_for_test();
                return false;
            }
            if (entry.expected_device != mem_handle::HOST_DEVICE && entry.handle.device() != entry.expected_device) {
                last_reject_ = moe_descriptor_reject_reason::DEVICE_MISMATCH;
                residency_diagnostics_record_stale_descriptor_device_mismatch_for_test();
                return false;
            }
            resolved_ptr resolved = entry.handle.resolve(entry.expected_device);
            if (!resolved.ptr) {
                last_reject_ = moe_descriptor_reject_reason::INVALID_HANDLE;
                residency_diagnostics_record_stale_descriptor_invalid_handle_for_test();
                return false;
            }
            if (entry.handle.stable_identity_hash() != entry.stable_identity_hash) {
                last_reject_ = moe_descriptor_reject_reason::IDENTITY_MISMATCH;
                residency_diagnostics_record_stale_descriptor_identity_mismatch_for_test();
                return false;
            }
            if (entry.expected_device != mem_handle::HOST_DEVICE && !resolved.on_device) {
                last_reject_ = moe_descriptor_reject_reason::DEVICE_MISMATCH;
                residency_diagnostics_record_stale_descriptor_device_mismatch_for_test();
                return false;
            }
            if (resolved.layout != entry.expected_layout) {
                last_reject_ = moe_descriptor_reject_reason::LAYOUT_MISMATCH;
                residency_diagnostics_record_stale_descriptor_layout_mismatch_for_test();
                return false;
            }
        }
        return true;
    }

    void replace_retained_handle_for_test(size_t idx, const mem_handle & handle) { entries_.at(idx).handle = handle; }

    void override_layout_for_test(size_t idx, ggml_layout_mode layout) { entries_.at(idx).expected_layout = layout; }

    void override_generation_for_test(size_t idx, uint64_t generation) { entries_.at(idx).generation = generation; }

  private:
    std::vector<moe_residency_descriptor_entry> entries_;
    mutable moe_descriptor_reject_reason        last_reject_ = moe_descriptor_reject_reason::NONE;
};

struct moe_residency_preflight_input {
    int              layer           = -1;
    int              device          = -1;
    size_t           required_bytes  = 0;
    ggml_layout_mode required_layout = GGML_LAYOUT_XMX_TILED;
    mem_handle       required_handle;
    residency_budget budget;
};

struct moe_residency_preflight_result {
    bool                    accepted                 = false;
    bool                    fallback_required        = true;
    bool                    optimized_launch_allowed = false;
    residency_reject_reason reason                   = residency_reject_reason::UNSUPPORTED;
    size_t                  bytes_requested          = 0;
    size_t                  largest_free_block       = 0;
    size_t                  retained_handle_count    = 0;
};

inline moe_residency_preflight_result evaluate_moe_residency_preflight_for_test(
    const moe_residency_preflight_input & input) {
    residency_request req;
    req.debug_name = "moe-preflight-test";
    req.device     = input.device;
    req.phase      = residency_phase::MOE_DECODE;
    residency_entry_request entry;
    entry.tensor_name = "moe-preflight";
    entry.layout      = input.required_layout;
    entry.bytes       = input.required_bytes;
    entry.role        = residency_role::MOE_GATE;
    if (input.required_handle.valid() || input.required_handle.has_stable_owner_identity()) {
        entry.handle         = input.required_handle;
        entry.require_handle = true;
    }
    req.entries.push_back(entry);

    residency_plan plan = evaluate_residency_request_for_test(req, input.budget);

    moe_residency_preflight_result result;
    result.accepted                 = plan.accepted;
    result.fallback_required        = !plan.accepted;
    result.optimized_launch_allowed = plan.accepted;
    result.reason                   = plan.reason;
    result.bytes_requested          = plan.bytes_requested;
    result.largest_free_block       = plan.largest_free_block;
    result.retained_handle_count    = plan.entries.size();
    return result;
}

struct moe_gate_up_pair {
    const ggml_tensor * up_weight            = nullptr;
    const ggml_tensor * gate_weight          = nullptr;
    const ggml_tensor * src1                 = nullptr;
    const ggml_tensor * ids                  = nullptr;
    const ggml_tensor * up_bias              = nullptr;
    const ggml_tensor * gate_bias            = nullptr;
    const ggml_tensor * down_bias            = nullptr;
    const ggml_tensor * down_weight          = nullptr;
    const ggml_tensor * moe_weights          = nullptr;
    ggml_tensor *       up_dst               = nullptr;
    ggml_tensor *       gate_dst             = nullptr;
    ggml_tensor *       up_biased            = nullptr;
    ggml_tensor *       gate_biased          = nullptr;
    ggml_tensor *       glu_dst              = nullptr;
    ggml_tensor *       down_dst             = nullptr;
    ggml_tensor *       down_biased          = nullptr;
    ggml_tensor *       weighted_dst         = nullptr;
    ggml_tensor *       down_sum_final       = nullptr;
    int                 up_index             = -1;
    int                 gate_index           = -1;
    int                 up_bias_index        = -1;
    int                 gate_bias_index      = -1;
    int                 glu_index            = -1;
    int                 down_index           = -1;
    int                 down_bias_index      = -1;
    int                 weighted_index       = -1;
    int                 down_sum_final_index = -1;
    enum ggml_glu_op    glu_op               = GGML_GLU_OP_SWIGLU;
};

struct moe_layer_grouped_route_row {
    size_t  entry = 0;
    int64_t iid1  = 0;
    int64_t id    = 0;
};

struct moe_layer_decode_role_plan {
    const ggml_tensor *                      weight = nullptr;
    ggml_layout_mode                         layout = GGML_LAYOUT_AOS;
    std::vector<int32_t>                     expert_ids;
    std::vector<mem_handle>                  handles;
    std::vector<moe_layer_grouped_route_row> rows;
    std::vector<sycl::event>                 ready_events;
    bool                                     full_table_static_handles = false;

    size_t entries() const { return expert_ids.size(); }

    bool has_handle_coverage() const { return handles.size() == expert_ids.size(); }
};

struct moe_layer_transient_ptr_table_slot {
    int32_t     expert_id = -1;
    mem_handle  handle;
    sycl::event ready_event;
    bool        has_ready_event = false;
};

struct moe_layer_decode_artifact_plan {
    const char *        role   = nullptr;
    const ggml_tensor * tensor = nullptr;
    mem_handle          handle;
    resolved_ptr        resolved;
};

struct moe_layer_persistent_tensor_descriptor {
    const char *        role   = nullptr;
    const ggml_tensor * tensor = nullptr;
    mem_handle          handle;
};

struct moe_layer_persistent_role_descriptor {
    const char *             role   = nullptr;
    const ggml_tensor *      weight = nullptr;
    ggml_layout_mode         layout = GGML_LAYOUT_AOS;
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
    moe_residency_descriptor               residency;

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
    std::vector<mem_handle>    residency_handles;

    bool                    down_eligible = false;
    moe_layer_executor_kind executor      = moe_layer_executor_kind::NONE;
    moe_layer_reject_reason reject_reason = moe_layer_reject_reason::NONE;

    bool has_gate_up_handles() const { return gate.has_handle_coverage() && up.has_handle_coverage(); }
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
    int32_t                                  expert_id    = -1;
    int                                      device       = mem_handle::HOST_DEVICE;
    ggml_layout_mode                         layout       = GGML_LAYOUT_AOS;
    moe_layer_route_residency                residency    = moe_layer_route_residency::MISSING;
    moe_route_phase                          phase        = moe_route_phase::DECODE;
    size_t                                   handle_index = SIZE_MAX;
    mem_handle                               handle;
    size_t                                   handle_hash = 0;
    std::vector<sycl::event>                 deps;
    std::vector<moe_layer_grouped_route_row> rows;
};

struct moe_layer_grouped_route_view {
    const ggml_tensor *                        weight               = nullptr;
    const char *                               role_name            = "unknown";
    int                                        submit_device        = -1;
    int64_t                                    top_k                = 0;
    int64_t                                    batch                = 0;
    size_t                                     entries              = 0;
    size_t                                     n_groups             = 0;
    size_t                                     device_rows          = 0;
    size_t                                     host_rows            = 0;
    size_t                                     missing_rows         = 0;
    size_t                                     layout_mismatch_rows = 0;
    size_t                                     max_rows_per_group   = 0;
    size_t                                     ready_events         = 0;
    std::vector<sycl::event>                   deps;
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
