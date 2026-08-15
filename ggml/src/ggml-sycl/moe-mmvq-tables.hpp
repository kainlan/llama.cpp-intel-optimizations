//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MOE_MMVQ_TABLES_HPP
#define GGML_SYCL_MOE_MMVQ_TABLES_HPP

#include "ggml.h"

// Single source of truth for batched-MoE MUL_MAT_ID (type, layout) coverage.
//
// Three tables and one invariant. Two describe what the executors can actually
// run; the third describes what the capability query is allowed to advertise:
//
//   moe_mmvq_batched_dispatch_supports_layout   -- mmvq_moe_batched_dispatch()
//   moe_mmvq_pair_glu_dispatch_supports_layout  -- the MXFP4 gate/up pair path
//   moe_mmvq_capability_supports_layout         -- ggml_sycl_moe_query_route_capability()
//
// INVARIANT: capability must be a SUBSET of the union of the executor tables.
//
// Advertising a pair no executor accepts is not a harmless optimism. The route
// is admitted, the executor then declines at submit, and the decline escapes as
// ggml_sycl_fallback_error -> GGML_STATUS_FAILED. ggml_backend_compare_graph_backend
// (ggml/src/ggml-backend.cpp) discards that status, so the harness compares a dst
// the backend never wrote and reports a numeric error instead of a refusal. A
// clean pre-submit refusal is strictly better than a false capability.
//
// The tables live here, outside any SYCL translation unit, so the invariant can
// be gated by a host-only test with no oneAPI toolchain and no GPU:
// tests/test-sycl-moe-mmvq-tables.cpp. Adding a type to one table without the
// other turns that test red.
//
// This header must stay free of SYCL and of ggml-sycl internals -- ggml.h only.

inline bool moe_mmvq_batched_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        // gx30 wave 1. AoS only: these dispatch through mmvq_submit_quant_aos_id(),
        // which instantiates the same generic kernel as the four above. No SoA or
        // packed variant exists for them, so do not widen the layout here without
        // adding the corresponding instantiation.
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_MXFP4:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED ||
                   layout == GGML_LAYOUT_MXFP4_I8 || layout == GGML_LAYOUT_MXFP4_DPAS ||
                   layout == GGML_LAYOUT_XMX_TILED;
        default:
            return false;
    }
}

// True when the type has an _id kernel family at all, whatever the layout. Lets
// a caller tell "this type is not covered" from "this type is covered but not in
// this layout" without restating the table.
inline bool moe_mmvq_batched_dispatch_supports_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// The MXFP4 gate/up pair path accepts the bundled tiled layout that the generic
// batched dispatch does not; without this table the invariant below would read
// GGML_LAYOUT_XMX_TILED_BUNDLE4 as an over-advertisement when it is in fact
// covered by a different executor.
inline bool moe_mmvq_pair_glu_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    if (type != GGML_TYPE_MXFP4) {
        return false;
    }
    return layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_XMX_TILED || layout == GGML_LAYOUT_XMX_TILED_BUNDLE4;
}

inline bool moe_mmvq_any_dispatch_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    return moe_mmvq_batched_dispatch_supports_layout(type, layout) ||
           moe_mmvq_pair_glu_dispatch_supports_layout(type, layout);
}

inline bool moe_mmvq_capability_supports_layout(enum ggml_type type, enum ggml_layout_mode layout) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return layout == GGML_LAYOUT_AOS;
        case GGML_TYPE_MXFP4:
            return layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED ||
                   layout == GGML_LAYOUT_MXFP4_I8 || layout == GGML_LAYOUT_MXFP4_DPAS ||
                   layout == GGML_LAYOUT_XMX_TILED || layout == GGML_LAYOUT_XMX_TILED_BUNDLE4;
        default:
            return false;
    }
}

#endif  // GGML_SYCL_MOE_MMVQ_TABLES_HPP
