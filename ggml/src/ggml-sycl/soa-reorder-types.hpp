//
// The one table of weight types whose AOS->SOA reorder exists (llama.cpp-76os).
//
// A SOA layout is only real for a type if the fill can materialize it: the
// device reorder (reorder_aos_to_soa_device, reorder_rows_to_soa in
// ggml-sycl.cpp) has a kernel for it. Three places must agree on that fact and
// used to keep a private copy of the list each:
//   - ggml_sycl_layout_supports_soa (ggml-sycl.cpp), which the runtime's
//     ggml_sycl_adjust_layout_for_tensor uses to clamp SOA to AOS;
//   - unified_cache::load_partial_rows (unified-cache.cpp), the partial-row fill;
//   - layout_policy::get_optimal (common.hpp), which the planner's
//     planner_default_device_layout takes its layout from.
// get_optimal's SOA default ("safe for all quantized types") was the one that
// disagreed: the planner planned SOA for IQ*/Q2_0 expert types while the runtime
// clamped them to AOS. The loaded layout is the answer, so the planner must not
// plan a layout the fill cannot materialize.
//
// Adding a type here is a claim that BOTH reorder switches have its kernel;
// tests/test-sycl-soa-reorder-types-source.py checks that they do.
//
// Pure C++ over the ggml_type enum: usable from a host-only test.
//
#pragma once

#include "ggml.h"

inline bool ggml_sycl_soa_reorder_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}
