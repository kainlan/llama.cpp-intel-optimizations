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

// The bytes a device copy of one expert in `layout` takes: what direct staging allocates for it. `expert_bytes` is the
// expert's own size, its tensor's bytes per expert (nb[2]), never the size of a copy of it, which a padded copy
// exceeds, so an expert sized from a copy would grow by the tail. For AOS the answer is the expert's bytes and the
// tail aos_staged_weight_bytes() adds, which the quantized kernels may read and which reads as zero. 0 for a layout
// this does not stage and for a type or width that names no weight, so a caller refuses rather than mints. The plan
// transaction mints the device storage of every expert it moves to the device, refilled or admitted, at this size and
// the cache checks the storage against it, so the two cannot disagree; a layout that stages another way (SOA's
// reorder) adds its own case here rather than at either caller.
inline size_t staged_expert_bytes(ggml_layout_mode layout, ggml_type type, int64_t ncols, size_t expert_bytes) {
    if (layout != GGML_LAYOUT_AOS || type >= GGML_TYPE_COUNT || ncols <= 0 || expert_bytes == 0) {
        return 0;
    }
    return aos_staged_weight_bytes(type, ncols, expert_bytes);
}

}  // namespace ggml_sycl
