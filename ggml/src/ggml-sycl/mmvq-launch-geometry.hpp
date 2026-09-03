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

//
// Small-N occupancy geometry for the Q8_0 SOA (`reorder`) MMVQ kernel
// (llama.cpp-6cgq).
//
// At a FIXED nrows, the SOA and COALESCED Q8_0 dispatch functions
// (mmvq.cpp: reorder_mul_mat_vec_q8_0_q8_1_sycl and
// coalesced_mul_mat_vec_q8_0_q8_1_sycl) build byte-identical nd_ranges: both
// derive `block_num_y` from the same `nrows`, both pad it with
// `mmvq_pad_rows_to_workgroups(block_num_y, 16)`, and both launch
// `padded_num_y / 16` work-groups of 16 sub-groups each -- one sub-group per
// output row either way. So the SOA-vs-COALESCED efficiency gap measured on
// llama.cpp-6cgq at a given N is NOT an occupancy difference between the two
// kernels; occupancy is identical. It is COALESCED's word-major tile layout
// reaching 100% cache-line utilization per fetch against SOA's block-major
// "reorder" layout's ~50% (documented at mul_mat_vec_q8_0_coalesced's
// definition in mmvq.cpp). That per-fetch waste only costs wall time when
// there is not enough OTHER concurrent work to hide the resulting memory
// latency -- and the amount of concurrent work is exactly the work-group
// count above, `nrows / subgroups_per_workgroup`, which shrinks as nrows
// shrinks. This is also why the gap is worse on the B70 (256 cores) than the
// B50 (128 cores) at the same N: the SAME work-group count fills half as much
// of the larger device.
//
// The fix below does not touch per-row memory access at all (that would mean
// rewriting the SOA byte layout to match COALESCED's word-major tiling, a far
// larger change). It only raises the work-group count for small nrows by
// spreading the SAME rows across MORE, smaller work-groups (fewer sub-groups
// per work-group) -- a launch-configuration-only change, safe under SYCL
// graph record/replay because it is a pure function of arguments already
// available at record time (nrows, ncols, device core count), and it costs
// nothing at large N because a work-group count already at or above the
// floor leaves `subgroups_per_workgroup` at today's 16, unchanged.
//
// `ncols` (K) matters too: for the SAME nrows, a longer per-row K loop gives
// each sub-group more of its own sequential memory transactions to overlap,
// which hides latency WITHIN a sub-group independently of how many other
// sub-groups are concurrently resident. Measured on llama.cpp-6cgq: at the
// same nrows=4096, the K=14336 "down"-projection shape (95%/92% of peak,
// SOA) already sits much closer to COALESCED than the K=4096 "q,o" shape
// (93%/85%) -- so the occupancy floor below is relaxed in proportion to how
// far ncols exceeds the K=4096 reference the deficit was profiled at,
// leaving large-K shapes at today's geometry even when their row count alone
// would otherwise cross the threshold.
struct mmvq_q8_0_soa_geometry_t {
    // Sub-groups bundled into one work-group; each sub-group computes
    // exactly one output row (see mul_mat_vec_q_reorder), so this is also
    // "rows handled per work-group". Always a divisor of 16, the value every
    // other reorder/coalesced MMVQ launch in mmvq.cpp still uses.
    int subgroups_per_workgroup;
    // K-splits per row. Always 1 today: geometry alone reaches the target
    // occupancy floor (see llama.cpp-6cgq comment), so split-K was not
    // implemented. Kept in the struct so a future split-K arm (a second
    // reduce kernel over unified-cache scratch) is an additive change to
    // callers rather than a signature break.
    int splits;
};

inline mmvq_q8_0_soa_geometry_t ggml_sycl_mmvq_q8_0_soa_geometry(const int nrows, const int ncols, const int n_cores) {
    constexpr int kDefaultSubgroupsPerWorkgroup = 16;    // today's fixed geometry; the ceiling, never exceeded
    constexpr int kReferenceCols                = 4096;  // K value the deficit was profiled at (llama.cpp-6cgq)
    constexpr int kOccupancyMultiplier          = 2;     // target >= 2x device cores worth of work-groups

    if (nrows <= 0 || n_cores <= 0) {
        // Nothing to size against (empty slice, or an unqueried device core
        // count) -- keep today's launch rather than guess.
        return { kDefaultSubgroupsPerWorkgroup, 1 };
    }

    const double k_relief        = (ncols > kReferenceCols) ? (double) kReferenceCols / (double) ncols : 1.0;
    int          occupancy_floor = (int) (kOccupancyMultiplier * n_cores * k_relief + 0.5);
    if (occupancy_floor < 1) {
        occupancy_floor = 1;
    }

    for (int subgroups = kDefaultSubgroupsPerWorkgroup; subgroups >= 1; subgroups /= 2) {
        const int padded_rows = mmvq_pad_rows_to_workgroups(nrows, subgroups);
        const int workgroups  = padded_rows / subgroups;
        if (workgroups >= occupancy_floor) {
            return { subgroups, 1 };
        }
    }
    return { 1, 1 };  // even the finest granularity available here cannot reach the floor -- best effort
}

}  // namespace ggml_sycl
