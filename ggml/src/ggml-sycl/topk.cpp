#include "topk.hpp"

#include "common.hpp"
#include "sycl-kernel-profiler.hpp"

#include <algorithm>

// GGML_OP_TOP_K for k above the MoE-gating kernel's limit (llama.cpp-1b6u).
// The path is chosen by ggml_sycl_top_k_select_path (topk.hpp); see there.
//
// Both paths compare ORDERED KEYS: the float's bits mapped so that unsigned
// order is float order (-inf lowest, +inf highest, NaN outside both), with
// ties broken by the lower column. That makes the selected set a pure
// function of the row, and the same set on either path.

// The ordering is total over bit patterns, not over float values: +0.0 ranks
// above -0.0, and a NaN ranks above +inf or below -inf by its sign bit. So
// "ties to the lower column" is exact only for equal bit patterns. CPU top_k
// leaves both ties and NaN unspecified, so this stays within its contract.
static inline uint32_t top_k_ordered_key(float v) {
    const uint32_t bits = sycl::bit_cast<uint32_t>(v);
    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

ggml_sycl_profile_label ggml_sycl_top_k_profile_label(sycl::queue & q,
                                                      const char *  name,
                                                      const char *  metadata,
                                                      size_t        bytes) {
    ggml_sycl_profile_label label{};
    if (!ggml_sycl_kernel_profile_enabled()) {
        return label;
    }
    label.name       = name;
    label.category   = "top_k";
    label.queue_kind = "compute";
    label.metadata   = metadata;
    label.device     = ggml_sycl_get_device_id_from_queue(q);
    label.bytes      = bytes;
    return label;
}

// ---------------------------------------------------------------------------
// BITONIC: one work-group per row sorts the whole padded row in SLM, by key
// descending then column ascending, and writes the first k columns.

static void top_k_bitonic_f32_i32_sycl(const float * x,
                                       int32_t *     dst,
                                       int64_t       ncols,
                                       int64_t       nrows,
                                       int64_t       k,
                                       sycl::queue * stream,
                                       int           device) {
    const size_t slm_bytes = ggml_sycl_top_k_bitonic_slm_bytes(ncols);
    GGML_ASSERT(slm_bytes <= ggml_sycl_info().devices[device].smpbo);
    const int ncols_pad = (int) (slm_bytes / (sizeof(uint32_t) + sizeof(int32_t)));

    const int max_wg = ggml_sycl_info().max_work_group_sizes[device];
    int       nth    = 1;
    while (nth * 2 <= ncols_pad && nth * 2 <= max_wg) {
        nth *= 2;
    }
    const int tasks = ncols_pad / nth;

    const sycl::nd_range<1> range(sycl::range<1>((size_t) nrows * nth), sycl::range<1>(nth));
    const size_t            bytes = (size_t) nrows * (ncols * sizeof(float) + k * sizeof(int32_t));

    const ggml_sycl_profile_label label =
        ggml_sycl_top_k_profile_label(*stream, "sycl.top_k.bitonic", "role=top_k;path=bitonic", bytes);
    ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> keys(sycl::range<1>(ncols_pad), cgh);
            sycl::local_accessor<int32_t, 1>  idxs(sycl::range<1>(ncols_pad), cgh);
            cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
                const int     tid   = (int) it.get_local_id(0);
                const int64_t row   = (int64_t) it.get_group(0);
                const float * x_row = x + row * ncols;

                // A pad slot carries the lowest key and a column >= ncols, so
                // it sorts after every real column, a real lowest key included.
                for (int i = 0; i < tasks; ++i) {
                    const int col = i * nth + tid;
                    keys[col]     = col < ncols ? top_k_ordered_key(x_row[col]) : 0u;
                    idxs[col]     = col;
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int kk = 2; kk <= ncols_pad; kk *= 2) {
                    for (int j = kk / 2; j > 0; j /= 2) {
                        for (int i = 0; i < tasks; ++i) {
                            const int col = i * nth + tid;
                            const int ixj = col ^ j;
                            if (ixj > col) {
                                const uint32_t ka      = keys[col];
                                const uint32_t kb      = keys[ixj];
                                const int32_t  ia      = idxs[col];
                                const int32_t  ib      = idxs[ixj];
                                // the pair is out of order when the element at ixj
                                // ranks first in a run sorted towards col
                                const bool     b_first = kb > ka || (kb == ka && ib < ia);
                                if (((col & kk) == 0) == b_first) {
                                    keys[col] = kb;
                                    keys[ixj] = ka;
                                    idxs[col] = ib;
                                    idxs[ixj] = ia;
                                }
                            }
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }

                for (int i = 0; i < tasks; ++i) {
                    const int col = i * nth + tid;
                    if (col < k) {
                        dst[row * k + col] = idxs[col];
                    }
                }
            });
        });
    });
}

// ---------------------------------------------------------------------------
// RADIX: an MSD radix select over the ordered keys, 8 bits per pass, finds the
// key T of the k-th largest column and how many columns equal to T to take
// (rank). Each row is split into `bpr` contiguous column ranges, one
// work-group each; per-range histograms are summed by a one-group-per-row
// select. The gather then writes every column with key > T and the first
// `rank` columns with key == T, both in column order, through an ordered
// compaction, so the output does not depend on scheduling.
//
// Scratch (uint32), all from the op pool:
//   state [nrows][4]           prefix, prefix mask, rank, unused
//   hist  [nrows][bpr][256]    per-range histograms of the current digit
//   cnt   [nrows][bpr][2]      per-range counts of key > T and key == T

static constexpr int     top_k_radix_block           = 256;
static constexpr int     top_k_radix_bins            = 256;
static constexpr int     top_k_radix_state_stride    = 4;
static constexpr int64_t top_k_radix_cols_per_group  = 2048;
static constexpr int64_t top_k_radix_groups_per_call = 256;

// The histogram kernel zeroes and stores one digit bin per work-item, and the
// select kernel scans one bin per work-item, so the work-group is exactly as
// wide as a digit has bins.
static_assert(top_k_radix_block == top_k_radix_bins, "TOP_K radix: one work-item per digit bin");

struct top_k_radix_geometry {
    int64_t bpr;   // column ranges (work-groups) per row
    int64_t span;  // columns per range
};

static top_k_radix_geometry top_k_radix_geometry_for(int64_t ncols, int64_t nrows) {
    const int64_t want = (ncols + top_k_radix_cols_per_group - 1) / top_k_radix_cols_per_group;
    const int64_t cap  = std::max<int64_t>(1, top_k_radix_groups_per_call / nrows);
    const int64_t bpr  = std::max<int64_t>(1, std::min(want, cap));
    return { bpr, (ncols + bpr - 1) / bpr };
}

static void top_k_radix_histogram(const float *              x,
                                  const uint32_t *           state,
                                  uint32_t *                 hist,
                                  int64_t                    ncols,
                                  int64_t                    nrows,
                                  const top_k_radix_geometry geo,
                                  int                        shift,
                                  bool                       first,
                                  sycl::queue *              stream) {
    const int64_t           bpr  = geo.bpr;
    const int64_t           span = geo.span;
    const sycl::nd_range<1> range(sycl::range<1>((size_t) (nrows * bpr) * top_k_radix_block),
                                  sycl::range<1>(top_k_radix_block));
    const size_t            bytes = (size_t) nrows * ncols * sizeof(float);

    const ggml_sycl_profile_label label = ggml_sycl_top_k_profile_label(*stream, "sycl.top_k.radix_histogram",
                                                                        "role=top_k;path=radix;kind=histogram", bytes);
    ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> local_hist(sycl::range<1>(top_k_radix_bins), cgh);
            cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
                const int     tid = (int) it.get_local_id(0);
                const int64_t g   = (int64_t) it.get_group(0);
                const int64_t row = g / bpr;
                const int64_t b   = g % bpr;

                local_hist[tid] = 0;
                it.barrier(sycl::access::fence_space::local_space);

                const uint32_t prefix = first ? 0u : state[row * top_k_radix_state_stride + 0];
                const uint32_t mask   = first ? 0u : state[row * top_k_radix_state_stride + 1];
                const float *  x_row  = x + row * ncols;
                const int64_t  begin  = b * span;
                const int64_t  end    = std::min<int64_t>(ncols, begin + span);
                for (int64_t c = begin + tid; c < end; c += top_k_radix_block) {
                    const uint32_t key = top_k_ordered_key(x_row[c]);
                    if ((key & mask) == prefix) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            bin(local_hist[(key >> shift) & (top_k_radix_bins - 1)]);
                        bin.fetch_add(1u);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                hist[(row * bpr + b) * top_k_radix_bins + tid] = local_hist[tid];
            });
        });
    });
}

static void top_k_radix_select(const uint32_t *           hist,
                               uint32_t *                 state,
                               int64_t                    nrows,
                               int64_t                    k,
                               const top_k_radix_geometry geo,
                               int                        shift,
                               bool                       first,
                               sycl::queue *              stream) {
    const int64_t           bpr = geo.bpr;
    const sycl::nd_range<1> range(sycl::range<1>((size_t) nrows * top_k_radix_block),
                                  sycl::range<1>(top_k_radix_block));
    const size_t            bytes = (size_t) nrows * bpr * top_k_radix_bins * sizeof(uint32_t);

    const ggml_sycl_profile_label label =
        ggml_sycl_top_k_profile_label(*stream, "sycl.top_k.radix_select", "role=top_k;path=radix;kind=select", bytes);
    ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
                const int     tid = (int) it.get_local_id(0);
                const int64_t row = (int64_t) it.get_group(0);
                // thread t owns digit 255 - t, so the inclusive scan counts the
                // columns at or above that digit
                const int     bin = top_k_radix_bins - 1 - tid;

                uint32_t count = 0;
                for (int64_t b = 0; b < bpr; ++b) {
                    count += hist[(row * bpr + b) * top_k_radix_bins + bin];
                }
                uint32_t *     st     = state + row * top_k_radix_state_stride;
                const uint32_t prefix = first ? 0u : st[0];
                const uint32_t mask   = first ? 0u : st[1];
                const uint32_t rank   = first ? (uint32_t) k : st[2];
                sycl::group_barrier(it.get_group());

                const uint32_t incl = sycl::inclusive_scan_over_group(it.get_group(), count, sycl::plus<uint32_t>());
                const uint32_t excl = incl - count;
                // exactly one digit holds the rank-th column: every column above
                // it is taken, and rank - excl are still needed inside it
                if (excl < rank && rank <= incl) {
                    st[0] = prefix | ((uint32_t) bin << shift);
                    st[1] = mask | ((uint32_t) (top_k_radix_bins - 1) << shift);
                    st[2] = rank - excl;
                }
            });
        });
    });
}

static void top_k_radix_count(const float *              x,
                              const uint32_t *           state,
                              uint32_t *                 cnt,
                              int64_t                    ncols,
                              int64_t                    nrows,
                              const top_k_radix_geometry geo,
                              sycl::queue *              stream) {
    const int64_t           bpr  = geo.bpr;
    const int64_t           span = geo.span;
    const sycl::nd_range<1> range(sycl::range<1>((size_t) (nrows * bpr) * top_k_radix_block),
                                  sycl::range<1>(top_k_radix_block));
    const size_t            bytes = (size_t) nrows * ncols * sizeof(float);

    const ggml_sycl_profile_label label =
        ggml_sycl_top_k_profile_label(*stream, "sycl.top_k.radix_count", "role=top_k;path=radix;kind=count", bytes);
    ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
                const int     tid = (int) it.get_local_id(0);
                const int64_t g   = (int64_t) it.get_group(0);
                const int64_t row = g / bpr;
                const int64_t b   = g % bpr;

                const uint32_t threshold = state[row * top_k_radix_state_stride + 0];
                const float *  x_row     = x + row * ncols;
                const int64_t  begin     = b * span;
                const int64_t  end       = std::min<int64_t>(ncols, begin + span);
                uint32_t       gt        = 0;
                uint32_t       eq        = 0;
                for (int64_t c = begin + tid; c < end; c += top_k_radix_block) {
                    const uint32_t key = top_k_ordered_key(x_row[c]);
                    gt += key > threshold ? 1u : 0u;
                    eq += key == threshold ? 1u : 0u;
                }
                gt = sycl::reduce_over_group(it.get_group(), gt, sycl::plus<uint32_t>());
                eq = sycl::reduce_over_group(it.get_group(), eq, sycl::plus<uint32_t>());
                if (tid == 0) {
                    cnt[(row * bpr + b) * 2 + 0] = gt;
                    cnt[(row * bpr + b) * 2 + 1] = eq;
                }
            });
        });
    });
}

static void top_k_radix_gather(const float *              x,
                               const uint32_t *           state,
                               const uint32_t *           cnt,
                               int32_t *                  dst,
                               int64_t                    ncols,
                               int64_t                    nrows,
                               int64_t                    k,
                               const top_k_radix_geometry geo,
                               sycl::queue *              stream) {
    const int64_t           bpr  = geo.bpr;
    const int64_t           span = geo.span;
    const sycl::nd_range<1> range(sycl::range<1>((size_t) (nrows * bpr) * top_k_radix_block),
                                  sycl::range<1>(top_k_radix_block));
    const size_t            bytes = (size_t) nrows * (ncols * sizeof(float) + k * sizeof(int32_t));

    const ggml_sycl_profile_label label =
        ggml_sycl_top_k_profile_label(*stream, "sycl.top_k.radix_gather", "role=top_k;path=radix;kind=gather", bytes);
    ggml_sycl_profile_submit(*stream, label, [&](sycl::queue & q) {
        return q.submit([&](sycl::handler & cgh) {
            cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
                const int     tid = (int) it.get_local_id(0);
                const int64_t g   = (int64_t) it.get_group(0);
                const int64_t row = g / bpr;
                const int64_t b   = g % bpr;

                const uint32_t threshold = state[row * top_k_radix_state_stride + 0];
                const uint32_t rank      = state[row * top_k_radix_state_stride + 2];
                const uint32_t n_gt      = (uint32_t) k - rank;

                // where this range's columns start in each half of the output
                uint32_t prev_gt = 0;
                uint32_t prev_eq = 0;
                for (int64_t b2 = tid; b2 < b; b2 += top_k_radix_block) {
                    prev_gt += cnt[(row * bpr + b2) * 2 + 0];
                    prev_eq += cnt[(row * bpr + b2) * 2 + 1];
                }
                uint32_t base_gt = sycl::reduce_over_group(it.get_group(), prev_gt, sycl::plus<uint32_t>());
                uint32_t base_eq = sycl::reduce_over_group(it.get_group(), prev_eq, sycl::plus<uint32_t>());

                const float * x_row   = x + row * ncols;
                int32_t *     dst_row = dst + row * k;
                const int64_t begin   = b * span;
                const int64_t end     = std::min<int64_t>(ncols, begin + span);
                // every work-item runs the same number of rounds, so the group
                // scans below see the whole group each time
                for (int64_t c0 = begin; c0 < end; c0 += top_k_radix_block) {
                    const int64_t  c     = c0 + tid;
                    const uint32_t key   = c < end ? top_k_ordered_key(x_row[c]) : 0u;
                    const uint32_t is_gt = c < end && key > threshold ? 1u : 0u;
                    const uint32_t is_eq = c < end && key == threshold ? 1u : 0u;

                    const uint32_t pre_gt =
                        sycl::exclusive_scan_over_group(it.get_group(), is_gt, sycl::plus<uint32_t>());
                    const uint32_t pre_eq =
                        sycl::exclusive_scan_over_group(it.get_group(), is_eq, sycl::plus<uint32_t>());
                    if (is_gt) {
                        dst_row[base_gt + pre_gt] = (int32_t) c;
                    }
                    if (is_eq && base_eq + pre_eq < rank) {
                        dst_row[n_gt + base_eq + pre_eq] = (int32_t) c;
                    }
                    base_gt += sycl::reduce_over_group(it.get_group(), is_gt, sycl::plus<uint32_t>());
                    base_eq += sycl::reduce_over_group(it.get_group(), is_eq, sycl::plus<uint32_t>());
                }
            });
        });
    });
}

void ggml_sycl_top_k_large(ggml_backend_sycl_context & ctx,
                           const float *               src,
                           int32_t *                   dst,
                           int64_t                     ncols,
                           int64_t                     nrows,
                           int64_t                     k,
                           ggml_sycl_top_k_path        path) {
    sycl::queue * stream = ctx.stream();

    if (path == GGML_SYCL_TOP_K_PATH_BITONIC) {
        top_k_bitonic_f32_i32_sycl(src, dst, ncols, nrows, k, stream, ctx.device);
        return;
    }
    GGML_ASSERT(path == GGML_SYCL_TOP_K_PATH_RADIX);
    GGML_ASSERT(top_k_radix_block <= ggml_sycl_info().max_work_group_sizes[ctx.device]);

    const top_k_radix_geometry geo           = top_k_radix_geometry_for(ncols, nrows);
    const size_t               state_elems   = (size_t) nrows * top_k_radix_state_stride;
    const size_t               hist_elems    = (size_t) (nrows * geo.bpr) * top_k_radix_bins;
    const size_t               cnt_elems     = (size_t) (nrows * geo.bpr) * 2;
    const size_t               scratch_elems = state_elems + hist_elems + cnt_elems;

    // The op pool hands out unified-cache SCRATCH sub-allocations owned by a
    // mem_handle, and returns them behind a marker event once the kernels
    // below have run, so the scratch is never reused under them.
    ggml_sycl_pool_alloc<uint32_t> scratch(ctx.pool(), scratch_elems);
    GGML_ASSERT(scratch.get() != nullptr && "TOP_K radix scratch: the op pool is exhausted");
    uint32_t * state = scratch.get();
    uint32_t * hist  = state + state_elems;
    uint32_t * cnt   = hist + hist_elems;

    for (int shift = 24; shift >= 0; shift -= 8) {
        const bool first = shift == 24;
        top_k_radix_histogram(src, state, hist, ncols, nrows, geo, shift, first, stream);
        top_k_radix_select(hist, state, nrows, k, geo, shift, first, stream);
    }
    top_k_radix_count(src, state, cnt, ncols, nrows, geo, stream);
    top_k_radix_gather(src, state, cnt, dst, ncols, nrows, k, geo, stream);
}
