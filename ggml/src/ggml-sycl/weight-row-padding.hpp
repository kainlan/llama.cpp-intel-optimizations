#pragma once

#include "ggml.h"
#include "presets.hpp"

#include <cstddef>
#include <cstdint>

namespace ggml_sycl {

// The bytes direct staging allocates for one AOS quantized weight of `ncols` columns whose rows take `base_size`
// bytes: the rows, plus a tail that completes the last row to a multiple of MATRIX_ROW_PADDING columns, which the
// quantized kernels may read. The stager sizes its allocation with this and the planner charges VRAM with it, so the
// plan and the zone cannot disagree about what one expert costs.
inline size_t aos_staged_weight_bytes(ggml_type type, int64_t ncols, size_t base_size) {
    if (!ggml_is_quantized(type)) {
        return base_size;
    }
    const int64_t rem = ncols % MATRIX_ROW_PADDING;
    if (rem == 0) {
        return base_size;
    }
    return base_size + ggml_row_size(type, MATRIX_ROW_PADDING - rem);
}

}  // namespace ggml_sycl
