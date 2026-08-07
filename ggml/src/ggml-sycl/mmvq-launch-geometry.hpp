//
// Launch geometry for the MMVQ sub-group-per-row kernels.
//
// The sub-group-per-row MMVQ family (`mul_mat_vec_q_reorder` and the Q4_0/Q8_0/
// MXFP4 coalesced kernels) gives dimension 2 a global range of
// `rows * WARP_SIZE` against a work-group of `num_subgroups * WARP_SIZE`, and
// derives its output row as `group_id * subgroup_range + subgroup_id`. SYCL's
// nd_range only accepts that pair when the global range divides evenly by the
// work-group size, and the Arc Pro B70 does not support non-uniform
// work-groups -- an indivisible pair throws
//
//     Non-uniform work-groups are not supported by the target device
//
// out of the dispatcher rather than running short. The launches used to pass
// the row count through `ceil_div(nrows, GGML_SYCL_MMV_Y)`, which looks like a
// rounding step but is a no-op: GGML_SYCL_MMV_Y is 1 (presets.hpp), so the
// requirement was really `nrows % 16 == 0`.
//
// `nrows` is a SLICE (`row_diff = row_high - row_low`), not the tensor's row
// count, and the slice parameter exists specifically for split-tensor support,
// so nothing upstream guarantees that multiple. Common transformer dimensions
// happen to be multiples of 16, which is the only reason this survived in
// practice (llama.cpp-99ke).
//
// The fix is the conventional SYCL pattern: pad the global range up to a whole
// number of work-groups and let the kernel body reject the rows past the end.
// Every kernel reached from these launches already opens with
// `if (row >= nrows) { return; }`, and `row` is sub-group-uniform, so the
// padded sub-groups exit before any memory traffic and before the sub-group
// reduction -- which is also why the early return cannot desynchronise a
// collective. Shapes that were already a multiple of 16 pad to themselves and
// launch byte-identical geometry.
//
// This header is deliberately dependency-free -- no SYCL header, no backend
// header -- for the same reason moe-scratch-admission.hpp is: the geometry is
// pure arithmetic, so it can be unit-tested on a host with no GPU and no SYCL
// runtime (see tests/test-mmvq-launch-geometry.cpp).
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#pragma once

namespace ggml_sycl {

// Round `rows` up to a whole number of work-groups of `subgroups_per_workgroup`
// rows each. Returns 0 for a non-positive row count, so an empty slice still
// launches nothing rather than a work-group of pure padding.
inline int mmvq_pad_rows_to_workgroups(const int rows, const int subgroups_per_workgroup) {
    if (rows <= 0) {
        return 0;
    }
    if (subgroups_per_workgroup <= 1) {
        return rows;
    }
    const int workgroups = (rows + subgroups_per_workgroup - 1) / subgroups_per_workgroup;
    return workgroups * subgroups_per_workgroup;
}

// The two properties the padding exists to provide, named so the host test
// asserts the contract instead of re-deriving the formula it is checking.

// Uniform work-groups: the property nd_range enforces and this device lacks.
inline bool mmvq_launch_is_uniform(const int padded_rows, const int subgroups_per_workgroup) {
    return subgroups_per_workgroup > 0 && padded_rows % subgroups_per_workgroup == 0;
}

// Coverage: padding must never drop a row the caller asked for.
inline bool mmvq_launch_covers_rows(const int padded_rows, const int rows) {
    return padded_rows >= rows;
}

}  // namespace ggml_sycl
