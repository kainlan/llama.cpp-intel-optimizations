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

// ---------------------------------------------------------------------------
// Column-split form of the Q8_0 matvec (SOA and coalesced layouts).
//
// The row form above is parallel over rows only: one sub-group per output row,
// walking the whole row in steps of `blocks_per_step` blocks (8 for the SOA
// kernel: vdr 2 ints x 32 lanes / 8 ints per block; one 32-block tile for the
// coalesced kernel). A matrix with few rows and long rows therefore runs on a
// handful of sub-groups while the card idles: 4 x 10240 is four sub-groups of
// 40 steps each.
//
// The column-split form keeps the row form's work-group (16 sub-groups) and
// gives each row `chunks` of them. Chunk c walks the row form's steps c,
// c + chunks, c + 2 * chunks, ...; each sub-group reduces its lanes as the row
// form does, the partials meet in the work-group's local memory (a work-group
// runs on one Xe-core), and one sub-group per row adds them in chunk order. A
// work-group holds 16 / chunks rows.
//
// When it is chosen (`mmvq_column_split_chunks`), from the launch geometry and
// one device fact, never from a tensor or model name:
//   * rows < compute_units. The row form gives each row one sub-group, a
//     sub-group runs as one hardware thread on one execution unit, and SYCL's
//     max_compute_units is the execution-unit count on these cards (256 on the
//     Arc Pro B70, 128 on the B50). With fewer rows than that, the row form
//     cannot give every execution unit even one thread; with at least that
//     many it can, and it is kept unchanged.
//   * at least two of the row form's steps per row, so every chunk owns at
//     least one step; fewer would only add idle sub-groups and a barrier.
// The chunk count is then the largest power of two that is at most 16 (the
// work-group's sub-groups, so a row never spans two work-groups) and at most
// the row's step count. Powers of two divide 16, so the rows per work-group
// are a whole number.
//
// Like the padding above, this is pure arithmetic with no SYCL dependency so
// tests/test-sycl-mmvq-column-split-rule.cpp can hold it on a host.

// The sub-groups of a column-split work-group: the row form's work-group, the
// size these launches are already known to run with.
constexpr int MMVQ_COLUMN_SPLIT_SUBGROUPS = 16;

// The row form's sub-group steps over one row: ceil(blocks / step).
inline int mmvq_column_split_steps(const int blocks_per_row, const int blocks_per_step) {
    if (blocks_per_row <= 0 || blocks_per_step <= 0) {
        return 0;
    }
    return (blocks_per_row + blocks_per_step - 1) / blocks_per_step;
}

// The dispatch rule. Returns 1 for the row form, otherwise the chunks per row.
// `compute_units` <= 0 (the device did not say) keeps the row form.
inline int mmvq_column_split_chunks(const int rows,
                                    const int blocks_per_row,
                                    const int blocks_per_step,
                                    const int compute_units) {
    if (rows <= 0 || compute_units <= 0 || rows >= compute_units) {
        return 1;
    }
    const int steps  = mmvq_column_split_steps(blocks_per_row, blocks_per_step);
    int       chunks = 1;
    while (chunks * 2 <= MMVQ_COLUMN_SPLIT_SUBGROUPS && chunks * 2 <= steps) {
        chunks *= 2;
    }
    return chunks;
}

inline int mmvq_column_split_rows_per_workgroup(const int chunks) {
    return chunks > 0 ? MMVQ_COLUMN_SPLIT_SUBGROUPS / chunks : 0;
}

// Whole work-groups covering `rows`; the last may hold padded rows, which the
// kernel computes nothing for and never writes.
inline int mmvq_column_split_workgroups(const int rows, const int chunks) {
    const int rows_per_workgroup = mmvq_column_split_rows_per_workgroup(chunks);
    if (rows <= 0 || rows_per_workgroup <= 0) {
        return 0;
    }
    return (rows + rows_per_workgroup - 1) / rows_per_workgroup;
}

// The first block a lane of chunk `chunk` reads, where `lane_block` is the
// lane's block within one step (lane / 4 for SOA, the lane for a coalesced
// tile). With one chunk this is the row form's own start.
inline int mmvq_column_split_first_block(const int chunk, const int lane_block, const int blocks_per_step) {
    return chunk * blocks_per_step + lane_block;
}

// The distance between a chunk's consecutive blocks. With one chunk this is
// the row form's own stride, one step.
inline int mmvq_column_split_block_stride(const int chunks, const int blocks_per_step) {
    return chunks * blocks_per_step;
}

}  // namespace ggml_sycl
