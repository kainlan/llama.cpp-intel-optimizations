// The one list of weight types the unified MUL_MAT kernel serves (llama.cpp-8ony).
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

}  // namespace ggml_sycl
