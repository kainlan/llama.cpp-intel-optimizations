// The one list of weight types the unified MUL_MAT kernel serves (llama.cpp-8ony), and the one list of weight types
// the layout policy can materialize COALESCED (llama.cpp-gldu).
//
// Header-only and free of SYCL, so both the dispatch router (dispatch.hpp, should_use_unified) and the zone planner's
// adapter (unified-cache.cpp) can include it. The planner needs the answer before any graph exists: it reserves the
// planned dequant buffers for a weight the unified kernel's oneDNN f16 route can draw, so a type added here without
// the planner seeing it would be drawn at run time and never planned. Keeping one list removes that drift.
#pragma once

#include "ggml.h"

namespace ggml_sycl {

// Unified kernel supports the block-quantized types Q4_0 and MXFP4. FP16/BF16 use oneDNN.
inline bool unified_kernel_serves_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_MXFP4:
            // TODO: Add Q8_0, Q6_K, Q4_K support to unified kernel
            return true;
        default:
            return false;  // FP16, BF16, F32, Q6_K, etc. use legacy path for now
    }
}

// The weight types the layout policy can materialize in the COALESCED layout. Add a type here only when its coalesced
// kernels exist. Read by is_coalesced_supported (common.hpp) and, through the complement below, by the zone planner.
inline bool coalesced_capable_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

// Whether a dense QUANTIZED MUL_MAT weight of this type takes the oneDNN f16 dequant arm as its PP route
// (llama.cpp-gldu). The caller has already established that the type is quantized and the operand a dense weight.
//
// The router sends an AOS-materialized quantized weight at PP batch to that arm (pick_kernel_for_layout,
// GGML_LAYOUT_AOS, ggml-sycl.cpp), and a type with no coalesced layout is materialized AOS: the IQ family and Q5_K have
// no SOA layout at all, and Q4_K is pinned AOS by layout_policy::get_optimal. For those types the arm is the planned
// route, not a fallback, so the planner reserves its buffers. A type with a coalesced layout is routed by its layout
// to a coalesced kernel (Q6_K), the oneDNN coalesced/SOA arm (Q8_0, which has its own unconditional plan mark) or the
// unified kernel (Q4_0 / MXFP4, which have the unified-kernel plan mark), so they are not claimed here.
//
// The planner cannot ask the router: it has no graph node, no batch and no resolved layout. This is a type-level
// superset the router can only narrow (an op the oneDNN PP scratch supplies, a batch within the MMVQ limit), and an
// unneeded reservation is bounded by one weight's f16 copy.
inline bool dense_pp_route_is_f16_dequant_arm(ggml_type type) {
    return !coalesced_capable_type(type);
}

}  // namespace ggml_sycl
