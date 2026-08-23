#include "convert.hpp"

#include "common.hpp"
#include "convert-esimd.hpp"
#include "dequantize.hpp"
#include "dmmv-coalesced-q4-0-layout.hpp"
#include "mem-handle.hpp"
#include "mem-ops.hpp"
#include "presets.hpp"

#include <utility>
#include <vector>

#if defined(__INTEL_LLVM_COMPILER)
#    if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#        include <sycl/ext/oneapi/bfloat16.hpp>
#        define GGML_SYCL_HAS_BF16
#    endif
#endif

// GGML_SYCL_ESIMD_DEQUANT=1 → enable ESIMD dequant (default OFF — standard SYCL
// auto-vectorizes to the same SIMD width as ESIMD for this memory-bound kernel,
// so the explicit ESIMD path does not win on Arc B580.  The env var exists for
// A/B comparison and future tuning.)
static bool g_esimd_dequant_enabled = []() {
    const char * env = std::getenv("GGML_SYCL_ESIMD_DEQUANT");
    if (env && env[0] == '1') {
#if GGML_SYCL_ESIMD_DEQUANT_AVAILABLE
        return true;
#endif
    }
    return false;
}();

static uint8_t * convert_alloc_device_scratch(size_t                  bytes,
                                              sycl::queue &           queue,
                                              const char *            cohort_id,
                                              ggml_sycl::mem_handle & owner) {
    owner = {};
    if (bytes == 0) {
        return nullptr;
    }

    ggml_sycl::alloc_request req{};
    req.queue                               = &queue;
    req.device                              = ggml_sycl_get_device_id_from_queue(queue);
    req.size                                = bytes;
    req.intent.role                         = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                     = ggml_sycl::runtime_category::COMPUTE;
    req.intent.cohort_id                    = cohort_id;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::SCRATCH;
    req.suppress_failure_log                = true;

    owner = ggml_sycl::unified_allocate(req);
    if (!owner.valid()) {
        return nullptr;
    }

    auto resolved = owner.resolve(req.device);
    if (!resolved || !resolved.ptr || !resolved.on_device) {
        owner = {};
        return nullptr;
    }
    return static_cast<uint8_t *>(resolved.ptr);
}

static sycl::event convert_copy_to_temp(sycl::queue &                 queue,
                                        uint8_t *                     temp,
                                        const ggml_sycl::mem_handle & temp_handle,
                                        const void *                  src,
                                        size_t                        bytes) {
    if (g_ggml_sycl_graph_recording) {
        return ggml_sycl_graph_safe_memcpy(queue, temp, src, bytes);
    }
    const int             queue_device = ggml_sycl_get_device_id_from_queue(queue);
    ggml_sycl::mem_handle src_handle   = ggml_sycl_memcpy_handle_for_raw_ptr(src, queue_device);
    return ggml_sycl::mem_copy_async(temp_handle, src_handle, bytes, queue);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx,
                             dst_t * __restrict__ y,
                             const int64_t            k,
                             const sycl::nd_item<3> & item_ct1) {
    const int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int64_t ib       = i / qk;         // block index
    const int64_t iqs      = (i % qk) / qr;  // quant index
    const int64_t iybs     = i - i % qk;     // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk / 2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void * __restrict__ vx,
                                  dst_t * __restrict__ y,
                                  const int64_t   k,
                                  dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1); });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q2_K(vx, y, item_ct1); });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q2_K(vx, y, item_ct1); });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q3_K(vx, y, item_ct1); });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q3_K(vx, y, item_ct1); });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb   = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q4_0(vx, y, nb32, item_ct1); });
    }
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp     = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                                           sycl::range<3>(1, 1, WARP_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
                         });
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_coalesced(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp     = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) * sycl::range<3>(1, 1, WARP_SIZE),
                                           sycl::range<3>(1, 1, WARP_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             dequantize_block_q4_0_coalesced(vx, y, k, item_ct1);
                         });
}

// COALESCED→row-major FP16 dequant for oneDNN PP path.
// Unlike the 1D dequantize_row_q4_0_sycl_coalesced (which uses flattened block
// indexing), this uses a 2D [nrows, tiles_per_row] grid with explicit per-row
// tile addressing so the output is guaranteed row-major.
template <typename dst_t>
static void dequantize_row_q4_0_sycl_coalesced_rowmajor(const void *    vx,
                                                        dst_t *         y,
                                                        const int       blocks_per_row,
                                                        const int       nrows,
                                                        dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    const int tiles_per_row = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    GGML_ASSERT(blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS == 0);

    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_q4_0_coalesced_rowmajor(vx, y, blocks_per_row, nrows, item);
        });
}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb   = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q4_1(vx, y, nb32, item_ct1); });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    dequantize_block_q4_K(vx, y, get_pointer(scale_local_acc), item_ct1);
                });
        });
    }
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb          = k / QK_K;
    const size_t  local_size  = 32;
    const size_t  global_size = nb * local_size;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                             dequantize_block_q4_K_reorder(vx, y, get_pointer(scale_local_acc), item_ct1, nb);
                         });
    });
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q5_K(vx, y, item_ct1); });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q5_K(vx, y, item_ct1); });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K(vx, y, item_ct1); });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
            [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K(vx, y, item_ct1); });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_q6_K_reorder(vx, y, item_ct1, nb); });
}

template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq1_s(vx, y, item_ct1, iq1s_grid_gpu); });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq1_m(vx, y, item_ct1, iq1s_grid_gpu); });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    dequantize_block_iq2_xxs(vx, y, item_ct1, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    dequantize_block_iq2_xs(vx, y, item_ct1, iq2xs_grid, ksigns_iq2xs, kmask_iq2xs);
                });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq2_s(vx, y, item_ct1); });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    dequantize_block_iq3_xxs(vx, y, item_ct1, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq3_s(vx, y, item_ct1, kmask_iq2xs, iq3s_grid); });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq4_xs(vx, y, item_ct1); });
        });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) { dequantize_block_iq4_nl(vx, y, item_ct1); });
        });
    }
}

template <typename dst_t>
static void dequantize_row_mxfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) { dequantize_block_mxfp4(vx, y, item_ct1); });
}

// NVFP4 packs four independently-scaled 16-element sub-blocks per block, so the
// generic <qk, qr, kernel> adapter cannot express it: QR_NVFP4 == 2 makes that
// adapter write element iqs+1's value at position iqs + QK_NVFP4/2.  One
// work-item per byte of qs, 32 per block, keeps the sub-block scale local.
static void dequantize_block_nvfp4_fp16(const void * vx, sycl::half * y, const sycl::nd_item<3> & item_ct1) {
    const int64_t ib  = item_ct1.get_group(2);
    const int     tid = item_ct1.get_local_id(2);
    const int     sub = tid / 8;
    const int     j   = tid % 8;

    const block_nvfp4 & xb = static_cast<const block_nvfp4 *>(vx)[ib];
    const uint8_t       q  = xb.qs[sub * 8 + j];
    const float         d  = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
    const int64_t       iy = ib * QK_NVFP4 + sub * QK_NVFP4_SUB + j;

    y[iy]     = sycl::half(d * kvalues_mxfp4[q & 0x0f]);
    y[iy + 8] = sycl::half(d * kvalues_mxfp4[q >> 4]);
}

static void dequantize_row_nvfp4_fp16_sycl(const void * vx, sycl::half * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int64_t nb = k / QK_NVFP4;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb * 32), sycl::range<3>(1, 1, 32)),
                         [=](sycl::nd_item<3> item_ct1) { dequantize_block_nvfp4_fp16(vx, y, item_ct1); });
}

template <typename src_t, typename dst_t>
static void convert_unary_nc(const void * __restrict__ vx,
                             dst_t * __restrict__ y,
                             const int64_t            ne00,
                             const int64_t            ne01,
                             const int64_t            ne02,
                             const int64_t            s01,
                             const int64_t            s02,
                             const int64_t            s03,
                             const sycl::nd_item<3> & item_ct1) {
    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x  = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    for (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_nc_sycl(const void * __restrict__ vx,
                                  dst_t * __restrict__ y,
                                  const int64_t   ne00,
                                  const int64_t   ne01,
                                  const int64_t   ne02,
                                  const int64_t   ne03,
                                  const int64_t   s01,
                                  const int64_t   s02,
                                  const int64_t   s03,
                                  dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    sycl::range<3> global_size(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_SIZE));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_size[0], SYCL_DEQUANTIZE_BLOCK_SIZE);
    sycl::range<3> workgroup_size(1, 1, downsized_workgroup);

    queue->parallel_for(
        sycl::nd_range<3>(global_size * workgroup_size, workgroup_size),
        [=](sycl::nd_item<3> item_ct1) { convert_unary_nc<src_t>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1); });
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}

// =============================================================================
// Q4_0 COALESCED REORDER KERNEL
// =============================================================================
// Reorders Q4_0 data from AoS (block_q4_0 structs) to a warp-coalesced layout
// where adjacent threads in a warp (32 threads) access adjacent memory addresses.
//
// Input layout (AoS - Array of Structures):
//   Each block_q4_0: [d:2 bytes][qs[0..15]:16 bytes] = 18 bytes per block
//   Blocks are stored contiguously: [block0][block1]...[blockN]
//
// Output layout (Coalesced):
//   Tiles of MMVQ_COALESCED_TILE_BLOCKS blocks where:
//   - qs bytes are word-major: word w of block b at offset [row_base + tile * tile_qs_bytes + w*stride + b*4]
//   - d values are stored AFTER ALL quants, block-sequential
// =============================================================================

// GPU kernel for Q4_0 AoS to Coalesced conversion
static void reorder_q4_0_aos_to_coalesced_kernel(const block_q4_0 * __restrict__ src,
                                                 uint8_t * __restrict__ dst,
                                                 const int                blocks_per_row,
                                                 const int                nrows,
                                                 const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORDS_PER_BLOCK = 4;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);  // thread within tile (warp lane)
    const int tile = item.get_group(1);     // Which tile

    if (row >= nrows) {
        return;
    }

    // Every offset comes from the Q4_0 addressing contract, so this writer and
    // the DMMV/MMVQ readers cannot drift apart. Deriving the row stride from
    // ggml_sycl_q8_0_coalesced_row_quants_bytes() here made the stride 2x too
    // large (Q8_0 carries 32 quant bytes per block, Q4_0 carries 16) and wrote
    // past the end of the coalesced allocation. See llama.cpp-szv8.
    const int64_t d_base = ggml_sycl::dmmv_coalesced_q4_0_scale_base(nrows, blocks_per_row);

    for (int block_in_tile = tid; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
        const int block_idx = tile * TILE_BLOCKS + block_in_tile;
        if (block_idx >= blocks_per_row) {
            continue;
        }

        // Source: AoS layout - each block_q4_0 is 18 bytes
        const block_q4_0 * src_block = &src[row * blocks_per_row + block_idx];

        // Copy qs bytes in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t word_offset =
                ggml_sycl::dmmv_coalesced_q4_0_qs_offset(row, blocks_per_row, block_idx, word * 4);
            memcpy(dst + word_offset, src_block->qs + word * 4, 4);
        }

        // Copy d value (scale) - contiguous after all quants
        *(sycl::half *) (dst + d_base +
                         ggml_sycl::dmmv_coalesced_q4_0_scale_index(row, blocks_per_row, block_idx) *
                             (int64_t) sizeof(sycl::half)) = src_block->d;
    }
}

// Host function to launch Q4_0 AoS to Coalesced reorder
void reorder_q4_0_aos_to_coalesced_sycl(const void *    src,
                                        void *          dst,
                                        int64_t         ne00,  // number of elements per row
                                        int64_t         ne01,  // number of rows
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK4_0 == 0);                            // Must be multiple of block size

    const int    blocks_per_row = ne00 / QK4_0;
    const int    nrows          = ne01;
    const int    tiles_per_row  = (blocks_per_row + MMVQ_COALESCED_TILE_BLOCKS - 1) / MMVQ_COALESCED_TILE_BLOCKS;
    const size_t total_blocks   = (size_t) blocks_per_row * (size_t) nrows;
    const size_t total_bytes    = total_blocks * sizeof(block_q4_0);

    // Always use a temp buffer to support in-place conversion safely
    ggml_sycl::mem_handle temp_handle;
    uint8_t * temp = convert_alloc_device_scratch(total_bytes, *stream, "convert_q4_0_coalesced", temp_handle);
    GGML_ASSERT(temp != nullptr);
    sycl::event copy_event = convert_copy_to_temp(*stream, temp, temp_handle, src, total_bytes);

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                reorder_q4_0_aos_to_coalesced_kernel((const block_q4_0 *) temp, (uint8_t *) dst, blocks_per_row, nrows,
                                                     item);
            });
    });

    std::vector<ggml_sycl::mem_handle> retained;
    retained.emplace_back(std::move(temp_handle));
    ggml_sycl::retain_handles_until_event(std::move(retained), std::move(convert_event));
}

// =============================================================================
// Q8_0 COALESCED REORDER KERNEL (AoS → Coalesced)
// =============================================================================

static void reorder_q8_0_aos_to_coalesced_kernel(const block_q8_0 * __restrict__ src,
                                                 uint8_t * __restrict__ dst,
                                                 const int                blocks_per_row,
                                                 const int                nrows,
                                                 const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS        = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int QS_BYTES_PER_BLOCK = QK8_0;  // 32 bytes
    constexpr int D_BYTES_PER_BLOCK  = sizeof(sycl::half);
    constexpr int WORDS_PER_BLOCK    = 8;
    constexpr int WORD_PLANE_STRIDE  = TILE_BLOCKS * 4;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);
    const int tile = item.get_group(1);

    if (row >= nrows) {
        return;
    }

    const int64_t row_quants_bytes   = (int64_t) ggml_sycl_q8_0_coalesced_row_quants_bytes(blocks_per_row);
    const int64_t total_quants_bytes = (int64_t) nrows * row_quants_bytes;
    const int64_t tile_qs_base = (int64_t) row * row_quants_bytes + (int64_t) tile * (TILE_BLOCKS * QS_BYTES_PER_BLOCK);

    for (int block_in_tile = tid; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
        const int block_idx = tile * TILE_BLOCKS + block_in_tile;
        if (block_idx >= blocks_per_row) {
            continue;
        }

        const block_q8_0 * src_block = &src[row * blocks_per_row + block_idx];

        // Copy qs bytes in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t word_offset = tile_qs_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst + word_offset, src_block->qs + word * 4, 4);
        }

        // Copy d value after all quants
        *(sycl::half *) (dst + total_quants_bytes + ((int64_t) row * blocks_per_row + block_idx) * D_BYTES_PER_BLOCK) =
            src_block->d;
    }
}

void reorder_q8_0_aos_to_coalesced_sycl(const void *    src,
                                        void *          dst,
                                        int64_t         ne00,
                                        int64_t         ne01,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK8_0 == 0);

    const int    blocks_per_row = ne00 / QK8_0;
    const int    nrows          = ne01;
    const int    tiles_per_row  = ggml_sycl_coalesced_fixed_tile_count(blocks_per_row);
    const size_t total_blocks   = (size_t) blocks_per_row * (size_t) nrows;
    const size_t total_bytes    = total_blocks * sizeof(block_q8_0);

    ggml_sycl::mem_handle temp_handle;
    uint8_t * temp = convert_alloc_device_scratch(total_bytes, *stream, "convert_q8_0_coalesced", temp_handle);
    GGML_ASSERT(temp != nullptr);
    sycl::event copy_event = convert_copy_to_temp(*stream, temp, temp_handle, src, total_bytes);

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                reorder_q8_0_aos_to_coalesced_kernel((const block_q8_0 *) temp, (uint8_t *) dst, blocks_per_row, nrows,
                                                     item);
            });
    });

    std::vector<ggml_sycl::mem_handle> retained;
    retained.emplace_back(std::move(temp_handle));
    ggml_sycl::retain_handles_until_event(std::move(retained), std::move(convert_event));
}

// =============================================================================
// MXFP4 COALESCED REORDER KERNEL (AoS → Coalesced)
// =============================================================================

static void reorder_mxfp4_aos_to_coalesced_kernel(const block_mxfp4 * __restrict__ src,
                                                  uint8_t * __restrict__ dst,
                                                  const int                blocks_per_row,
                                                  const int                nrows,
                                                  const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS        = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int QS_BYTES_PER_BLOCK = QK_MXFP4 / 2;  // 16 bytes
    constexpr int WORDS_PER_BLOCK    = 4;
    constexpr int WORD_PLANE_STRIDE  = TILE_BLOCKS * 4;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);
    const int tile = item.get_group(1);

    if (row >= nrows) {
        return;
    }

    const int64_t row_quants_bytes   = (int64_t) blocks_per_row * QS_BYTES_PER_BLOCK;
    const int64_t total_quants_bytes = (int64_t) nrows * row_quants_bytes;
    const int64_t tile_qs_base = (int64_t) row * row_quants_bytes + (int64_t) tile * (TILE_BLOCKS * QS_BYTES_PER_BLOCK);

    for (int block_in_tile = tid; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
        const int block_idx = tile * TILE_BLOCKS + block_in_tile;
        if (block_idx >= blocks_per_row) {
            continue;
        }

        const block_mxfp4 * src_block = &src[row * blocks_per_row + block_idx];

        // Copy qs bytes in word-major order
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t word_offset = tile_qs_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst + word_offset, src_block->qs + word * 4, 4);
        }

        // Copy exponent after all quants
        dst[total_quants_bytes + (int64_t) row * blocks_per_row + block_idx] = src_block->e;
    }
}

void reorder_mxfp4_aos_to_coalesced_sycl(const void *    src,
                                         void *          dst,
                                         int64_t         ne00,
                                         int64_t         ne01,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK_MXFP4 == 0);
    GGML_ASSERT((ne00 / QK_MXFP4) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int    blocks_per_row = ne00 / QK_MXFP4;
    const int    nrows          = ne01;
    const int    tiles_per_row  = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const size_t total_blocks   = (size_t) blocks_per_row * (size_t) nrows;
    const size_t total_bytes    = total_blocks * sizeof(block_mxfp4);

    ggml_sycl::mem_handle temp_handle;
    uint8_t * temp = convert_alloc_device_scratch(total_bytes, *stream, "convert_mxfp4_coalesced", temp_handle);
    GGML_ASSERT(temp != nullptr);
    sycl::event copy_event = convert_copy_to_temp(*stream, temp, temp_handle, src, total_bytes);

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                reorder_mxfp4_aos_to_coalesced_kernel((const block_mxfp4 *) temp, (uint8_t *) dst, blocks_per_row,
                                                      nrows, item);
            });
    });

    std::vector<ggml_sycl::mem_handle> retained;
    retained.emplace_back(std::move(temp_handle));
    ggml_sycl::retain_handles_until_event(std::move(retained), std::move(convert_event));
}

// =============================================================================
// Q6_K COALESCED REORDER KERNEL (AoS → Coalesced)
// =============================================================================

static void reorder_q6_k_aos_to_coalesced_kernel(const block_q6_K * __restrict__ src,
                                                 uint8_t * __restrict__ dst,
                                                 const int                blocks_per_row,
                                                 const int                nrows,
                                                 const int                tiles_per_row,
                                                 const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;
    constexpr int QL_WORDS          = (QK_K / 2) / 4;   // 32 words
    constexpr int QH_WORDS          = (QK_K / 4) / 4;   // 16 words
    constexpr int SC_WORDS          = (QK_K / 16) / 4;  // 4 words

    constexpr int QL_TILE_BYTES = TILE_BLOCKS * (QK_K / 2);
    constexpr int QH_TILE_BYTES = TILE_BLOCKS * (QK_K / 4);
    constexpr int SC_TILE_BYTES = TILE_BLOCKS * (QK_K / 16);
    constexpr int TILE_TOTAL    = QL_TILE_BYTES + QH_TILE_BYTES + SC_TILE_BYTES;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);
    const int tile = item.get_group(1);

    if (row >= nrows) {
        return;
    }

    const int64_t row_quants_bytes   = (int64_t) tiles_per_row * TILE_TOTAL;
    const int64_t total_quants_bytes = (int64_t) nrows * row_quants_bytes;

    const int64_t tile_base = (int64_t) row * row_quants_bytes + (int64_t) tile * TILE_TOTAL;
    const int64_t ql_base   = tile_base;
    const int64_t qh_base   = ql_base + QL_TILE_BYTES;
    const int64_t sc_base   = qh_base + QH_TILE_BYTES;

    for (int block_in_tile = tid; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
        const int block_idx = tile * TILE_BLOCKS + block_in_tile;
        if (block_idx >= blocks_per_row) {
            continue;
        }

        const block_q6_K * src_block = &src[row * blocks_per_row + block_idx];

        // ql (128 bytes)
        for (int word = 0; word < QL_WORDS; ++word) {
            const int64_t dst_offset = ql_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst + dst_offset, src_block->ql + word * 4, 4);
        }

        // qh (64 bytes)
        for (int word = 0; word < QH_WORDS; ++word) {
            const int64_t dst_offset = qh_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst + dst_offset, src_block->qh + word * 4, 4);
        }

        // scales (16 bytes)
        for (int word = 0; word < SC_WORDS; ++word) {
            const int64_t dst_offset = sc_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst + dst_offset, src_block->scales + word * 4, 4);
        }

        // d after all quants
        *(sycl::half *) (dst + total_quants_bytes + ((int64_t) row * blocks_per_row + block_idx) * sizeof(sycl::half)) =
            src_block->d;
    }
}

void reorder_q6_k_aos_to_coalesced_sycl(const void *    src,
                                        void *          dst,
                                        int64_t         ne00,
                                        int64_t         ne01,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    GGML_ASSERT((ne00 / QK_K) % MMVQ_COALESCED_TILE_BLOCKS == 0);

    const int    blocks_per_row = ne00 / QK_K;
    const int    nrows          = ne01;
    const int    tiles_per_row  = blocks_per_row / MMVQ_COALESCED_TILE_BLOCKS;
    const size_t total_blocks   = (size_t) blocks_per_row * (size_t) nrows;
    const size_t total_bytes    = total_blocks * sizeof(block_q6_K);

    ggml_sycl::mem_handle temp_handle;
    uint8_t * temp = convert_alloc_device_scratch(total_bytes, *stream, "convert_q6_k_coalesced", temp_handle);
    GGML_ASSERT(temp != nullptr);
    sycl::event copy_event = convert_copy_to_temp(*stream, temp, temp_handle, src, total_bytes);

    sycl::event convert_event = stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(copy_event);
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                reorder_q6_k_aos_to_coalesced_kernel((const block_q6_K *) temp, (uint8_t *) dst, blocks_per_row, nrows,
                                                     tiles_per_row, item);
            });
    });

    std::vector<ggml_sycl::mem_handle> retained;
    retained.emplace_back(std::move(temp_handle));
    ggml_sycl::retain_handles_until_event(std::move(retained), std::move(convert_event));
}

// =============================================================================
// Q6_K VARIABLE TILE COALESCED REORDER (CPU-side, for arbitrary block counts)
// =============================================================================
// Uses power-of-2 tile decomposition (largest first, max 32) to handle
// arbitrary block counts like 56 (32+16+8). Each tile uses word-major layout.

void reorder_q6_K_variable_tile(const block_q6_K * src,
                                uint8_t *          dst,
                                int64_t            nrows,
                                int64_t            blocks_per_row,
                                int64_t            row_stride) {
    const int num_tiles = tile_count(blocks_per_row);

    for (int64_t row = 0; row < nrows; row++) {
        uint8_t * row_dst   = dst + row * row_stride;
        int       block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_size = tile_size_at(blocks_per_row, tile);

            // Reorder ql: word-major (32 words of 4 bytes each per block)
            for (int word = 0; word < 32; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.ql[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder qh: word-major (16 words of 4 bytes each per block)
            for (int word = 0; word < 16; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.qh[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder scales: word-major (4 words of 4 bytes each per block)
            for (int word = 0; word < 4; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.scales[word * 4], 4);
                    row_dst += 4;
                }
            }

            block_idx += tile_size;
        }
    }
}

// =============================================================================
// AoS -> SoA conversion (direct, non-coalesced)
// =============================================================================

void reorder_q4_0_aos_to_soa_sycl(const void * src, void * dst, int64_t ne00, int64_t ne01, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK4_0 == 0);
    const int64_t blocks_per_row = ne00 / QK4_0;
    const int64_t nblocks        = blocks_per_row * ne01;

    const block_q4_0 * x      = (const block_q4_0 *) src;
    uint8_t *          qs_ptr = (uint8_t *) dst;
    sycl::half *       d_ptr  = (sycl::half *) (qs_ptr + nblocks * (QK4_0 / 2));

    stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        for (int j = 0; j < QK4_0 / 2; ++j) {
            qs_ptr[ib * (QK4_0 / 2) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
}

void reorder_q8_0_aos_to_soa_sycl(const void * src, void * dst, int64_t ne00, int64_t ne01, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK8_0 == 0);
    const int64_t blocks_per_row = ne00 / QK8_0;
    const int64_t nblocks        = blocks_per_row * ne01;

    const block_q8_0 * x      = (const block_q8_0 *) src;
    uint8_t *          qs_ptr = (uint8_t *) dst;
    sycl::half *       d_ptr  = (sycl::half *) (qs_ptr + nblocks * QK8_0);

    stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        for (int j = 0; j < QK8_0; ++j) {
            qs_ptr[ib * QK8_0 + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
}

void reorder_q4_k_aos_to_soa_sycl(const void * src, void * dst, int64_t nblocks, dpct::queue_ptr stream) {
    const block_q4_K * x          = (const block_q4_K *) src;
    uint8_t *          qs_ptr     = (uint8_t *) dst;
    uint8_t *          scales_ptr = qs_ptr + (QK_K / 2) * nblocks;
    sycl::half2 *      dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
}

void reorder_q6_k_aos_to_soa_sycl(const void * src, void * dst, int64_t nblocks, dpct::queue_ptr stream) {
    const block_q6_K * x          = (const block_q6_K *) src;
    uint8_t *          ql_ptr     = (uint8_t *) dst;
    uint8_t *          qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    uint8_t *          scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half *       d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        for (int j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
}

void reorder_mxfp4_aos_to_soa_sycl(const void * src, void * dst, int64_t ne00, int64_t ne01, dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK_MXFP4 == 0);
    const int64_t blocks_per_row = ne00 / QK_MXFP4;
    const int64_t nblocks        = blocks_per_row * ne01;

    const block_mxfp4 * x      = (const block_mxfp4 *) src;
    uint8_t *           qs_ptr = (uint8_t *) dst;
    uint8_t *           e_ptr  = qs_ptr + nblocks * (QK_MXFP4 / 2);

    stream->parallel_for(nblocks, [=](auto i) {
        const int64_t ib = i;
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            qs_ptr[ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
        }
        e_ptr[ib] = x[ib].e;
    });
}

// =============================================================================
// Q4_0 COALESCED to SoA REVERSE CONVERSION (for compatibility/debugging)
// =============================================================================
// This reverses the coalesced layout back to SoA layout if needed.
// Not typically used in the hot path, but useful for testing.

static void reorder_q4_0_coalesced_to_soa_kernel(const uint8_t * __restrict__ src,
                                                 uint8_t * __restrict__ dst_qs,
                                                 sycl::half * __restrict__ dst_d,
                                                 const int                blocks_per_row,
                                                 const int                nrows,
                                                 const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS        = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int QS_BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int D_BYTES_PER_BLOCK  = sizeof(sycl::half);
    constexpr int WORDS_PER_BLOCK    = 4;
    constexpr int WORD_PLANE_STRIDE  = TILE_BLOCKS * 4;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);
    const int tile = item.get_group(1);

    if (row >= nrows) {
        return;
    }

    // Coalesced layout: all quants first, then all d values
    const int64_t row_quants_bytes   = (int64_t) blocks_per_row * QS_BYTES_PER_BLOCK;
    const int64_t total_quants_bytes = (int64_t) nrows * row_quants_bytes;
    const int64_t tile_qs_base = (int64_t) row * row_quants_bytes + (int64_t) tile * (TILE_BLOCKS * QS_BYTES_PER_BLOCK);

    for (int block_in_tile = tid; block_in_tile < TILE_BLOCKS; block_in_tile += WARP_SIZE) {
        const int block_idx = tile * TILE_BLOCKS + block_in_tile;
        if (block_idx >= blocks_per_row) {
            continue;
        }

        // Destination: SoA layout - use int64_t for offsets
        // qs: all quants contiguous, then all d values
        const int64_t dst_qs_offset =
            (int64_t) row * blocks_per_row * QS_BYTES_PER_BLOCK + block_idx * QS_BYTES_PER_BLOCK;
        const int64_t dst_d_idx = (int64_t) row * blocks_per_row + block_idx;

        // Read word-major qs and write to contiguous SoA
        for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
            const int64_t word_offset = tile_qs_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(dst_qs + dst_qs_offset + word * 4, src + word_offset, 4);
        }

        // Read d value and write to d array
        dst_d[dst_d_idx] = *(const sycl::half *) (src + total_quants_bytes + dst_d_idx * D_BYTES_PER_BLOCK);
    }
}

// Public wrapper: COALESCED Q4_0 → row-major FP16 (for oneDNN PP)
// Dispatches to ESIMD kernel when available (and GGML_SYCL_ESIMD_DEQUANT != 0),
// otherwise falls back to standard-SYCL implementation.
void dequantize_row_q4_0_coalesced_to_fp16_rowmajor(const void *    src,
                                                    sycl::half *    dst,
                                                    int             blocks_per_row,
                                                    int             nrows,
                                                    dpct::queue_ptr stream) {
    if (g_esimd_dequant_enabled) {
        dequantize_row_q4_0_coalesced_to_fp16_rowmajor_esimd(src, dst, blocks_per_row, nrows, stream);
        return;
    }
    dequantize_row_q4_0_sycl_coalesced_rowmajor(src, dst, blocks_per_row, nrows, stream);
}

static void dequantize_block_q8_0_soa_rowmajor(const void * __restrict__ vx,
                                               sycl::half * __restrict__ yy,
                                               const int                blocks_per_row,
                                               const int                total_blocks,
                                               const sycl::nd_item<3> & item) {
    const int64_t block_i      = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    const int64_t row          = block_i / blocks_per_row;
    const int64_t block        = block_i - row * blocks_per_row;
    if (block >= blocks_per_row) {
        return;
    }

    const int8_t *    qs = static_cast<const int8_t *>(vx);
    const sycl::half * d  = reinterpret_cast<const sycl::half *>(qs + total_blocks * QK8_0);
    sycl::half *       y  = yy + block_i * QK8_0;
    const float        df = static_cast<float>(d[block_i]);

#pragma unroll
    for (int j = 0; j < QK8_0; ++j) {
        y[j] = sycl::half(df * static_cast<float>(qs[block_i * QK8_0 + j]));
    }
}

static void dequantize_block_q8_0_coalesced_rowmajor(const void * __restrict__ vx,
                                                     sycl::half * __restrict__ yy,
                                                     const int                blocks_per_row,
                                                     const int                nrows,
                                                     const sycl::nd_item<2> & item) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int WORDS_PER_BLOCK   = 8;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const int row  = item.get_global_id(0);
    const int tid  = item.get_local_id(1);
    const int tile = item.get_group(1);
    if (row >= nrows) {
        return;
    }

    const uint8_t * src                 = static_cast<const uint8_t *>(vx);
    const int64_t   row_quants_bytes    = static_cast<int64_t>(ggml_sycl_q8_0_coalesced_row_quants_bytes(blocks_per_row));
    const int64_t   total_quants_bytes  = static_cast<int64_t>(nrows) * row_quants_bytes;
    const int64_t   tile_qs_base        = static_cast<int64_t>(row) * row_quants_bytes +
                                    static_cast<int64_t>(tile) * MMVQ_COALESCED_TILE_BYTES_Q8_0;
    const int       block_in_tile       = tid;
    const int       block_idx           = tile * TILE_BLOCKS + block_in_tile;
    if (block_idx >= blocks_per_row) {
        return;
    }

    const int64_t      block_i = static_cast<int64_t>(row) * blocks_per_row + block_idx;
    const sycl::half * d       = reinterpret_cast<const sycl::half *>(src + total_quants_bytes);
    sycl::half *       y       = yy + block_i * QK8_0;
    const float        df      = static_cast<float>(d[block_i]);

#pragma unroll
    for (int word = 0; word < WORDS_PER_BLOCK; ++word) {
        const int64_t word_offset = tile_qs_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
        const int8_t * q          = reinterpret_cast<const int8_t *>(src + word_offset);
        y[word * 4 + 0]           = sycl::half(df * static_cast<float>(q[0]));
        y[word * 4 + 1]           = sycl::half(df * static_cast<float>(q[1]));
        y[word * 4 + 2]           = sycl::half(df * static_cast<float>(q[2]));
        y[word * 4 + 3]           = sycl::half(df * static_cast<float>(q[3]));
    }
}

void dequantize_row_q8_0_soa_to_fp16_rowmajor(const void *    src,
                                              sycl::half *    dst,
                                              int             blocks_per_row,
                                              int             nrows,
                                              dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    const int total_blocks = nrows * blocks_per_row;
    constexpr int WG_SIZE  = 256;
    const int n_wgs        = (total_blocks + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            const int64_t block_i = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
            if (block_i < total_blocks) {
                dequantize_block_q8_0_soa_rowmajor(src, dst, blocks_per_row, total_blocks, item);
            }
        });
}

void dequantize_row_q8_0_coalesced_to_fp16_rowmajor(const void *    src,
                                                    sycl::half *    dst,
                                                    int             blocks_per_row,
                                                    int             nrows,
                                                    dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    const int tiles_per_row = ggml_sycl_coalesced_fixed_tile_count(blocks_per_row);
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_block_q8_0_coalesced_rowmajor(src, dst, blocks_per_row, nrows, item);
        });
}

// SOA→row-major FP16 dequant for oneDNN PP path.
// SOA has sequential qs reads (no tile interleaving) — much faster than COALESCED.
// Dispatches to ESIMD kernel when available (and GGML_SYCL_ESIMD_DEQUANT != 0),
// otherwise falls back to standard-SYCL implementation.
void dequantize_row_q4_0_soa_to_fp16_rowmajor(const void *    src,
                                              sycl::half *    dst,
                                              int             blocks_per_row,
                                              int             nrows,
                                              dpct::queue_ptr stream) {
    if (g_esimd_dequant_enabled) {
        dequantize_row_q4_0_soa_to_fp16_rowmajor_esimd(src, dst, blocks_per_row, nrows, stream);
        return;
    }

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    const int     total_blocks = nrows * blocks_per_row;
    constexpr int WG_SIZE      = 256;
    const int     n_wgs        = (total_blocks + WG_SIZE - 1) / WG_SIZE;

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) { dequantize_block_q4_0_soa_rowmajor(src, dst, blocks_per_row, nrows, item); });
}

static void dequantize_tile_mxfp4_soa_rowmajor(const void * __restrict__ vx,
                                               sycl::half * __restrict__ yy,
                                               const int                blocks_per_row,
                                               const int                tiles_per_row,
                                               const int                nrows,
                                               const sycl::nd_item<3> & item) {
    (void) nrows;
    constexpr int BLOCKS_PER_TILE = QK_K / QK_MXFP4;

    const int64_t group = item.get_group(2);
    const int     row   = group / tiles_per_row;
    const int     tile  = group - row * tiles_per_row;
    const int     tid   = item.get_local_id(2);
    const int     il    = tid / BLOCKS_PER_TILE;
    const int     ib    = tid - il * BLOCKS_PER_TILE;
    const int     block = tile * BLOCKS_PER_TILE + ib;

    if (block >= blocks_per_row) {
        return;
    }

    const int64_t nblocks = static_cast<int64_t>(nrows) * blocks_per_row;
    const int64_t block_i = static_cast<int64_t>(row) * blocks_per_row + block;

    const uint8_t * qs = static_cast<const uint8_t *>(vx);
    const uint8_t * e  = qs + nblocks * (QK_MXFP4 / 2);
    const uint8_t * q4 = qs + block_i * (QK_MXFP4 / 2) + 4 * il;
    sycl::half *    y  = yy + block_i * QK_MXFP4 + 4 * il;

    const float d = sycl_e8m0_to_fp32_half(e[block_i]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q     = q4[j];
        y[j]                = sycl::half(d * kvalues_mxfp4[q & 0xf]);
        y[j + QK_MXFP4 / 2] = sycl::half(d * kvalues_mxfp4[q >> 4]);
    }
}

void dequantize_row_mxfp4_soa_to_fp16_rowmajor(const void *    src,
                                               sycl::half *    dst,
                                               int             blocks_per_row,
                                               int             nrows,
                                               dpct::queue_ptr stream) {
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    constexpr int BLOCKS_PER_TILE = QK_K / QK_MXFP4;
    constexpr int WG_SIZE         = 32;
    const int     tiles_per_row   = (blocks_per_row + BLOCKS_PER_TILE - 1) / BLOCKS_PER_TILE;
    const int     total_tiles     = nrows * tiles_per_row;

    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, total_tiles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             dequantize_tile_mxfp4_soa_rowmajor(src, dst, blocks_per_row, tiles_per_row, nrows, item);
                         });
}

// SOA MXFP4 -> WOQ (weight-only-quantized) repack: de-interleaves the intra-
// block j/j+16 nibble packing into SEQUENTIAL nibble order and transposes the
// per-expert row-major source into oneDNN's {K,N} nibble / {K/QK_MXFP4,N}
// e8m0-scale layout (perf-recovery epic, track C; nibble order + scale layout
// confirmed by the C1 spike, llama.cpp-4m9p comment c-a5by). 4-bit->4-bit: the
// nibble VALUES are the raw e2m1 codes, unchanged -- only their position moves.
//
// Source (SOA, per expert; the inverse of what dequantize_tile_mxfp4_soa_rowmajor
// above decodes): qs plane is nblocks*(QK_MXFP4/2) bytes, nblocks = nrows *
// blocks_per_row, block_i = row * blocks_per_row + block; within a block's
// QK_MXFP4/2 bytes, byte j holds element j (low nibble) and element j+QK_MXFP4/2
// (high nibble). The e8m0 plane follows: nblocks bytes at qs + nblocks*(QK_MXFP4/2).
//
// Destination: nibbles for weights {K,N} strides {N,1} -- element index
// el = k*N + n, dst byte el/2, low nibble if el even else high (K = blocks_per_row
// * QK_MXFP4, N = nrows). Scales {K/QK_MXFP4,N} strides {N,1}:
// dst_scales[(k/QK_MXFP4)*N + n] = src_e[n*blocks_per_row + k/QK_MXFP4].
static void repack_mxfp4_soa_to_woq_nibbles_kernel(const uint8_t * __restrict__ qs,
                                                   uint8_t * __restrict__ dst_nibbles,
                                                   int                      blocks_per_row,
                                                   int                      N,
                                                   int64_t                  total_dst_bytes,
                                                   const sycl::nd_item<3> & item) {
    const int64_t db = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (db >= total_dst_bytes) {
        return;
    }

    auto src_nibble = [=](int64_t el) -> uint8_t {
        const int64_t k       = el / N;
        const int64_t n       = el - k * N;
        const int64_t b       = k / QK_MXFP4;
        const int64_t k_local = k - b * QK_MXFP4;
        const int64_t block_i = n * blocks_per_row + b;
        const bool    lo      = k_local < QK_MXFP4 / 2;
        const uint8_t byte    = qs[block_i * (QK_MXFP4 / 2) + (lo ? k_local : k_local - QK_MXFP4 / 2)];
        return lo ? (byte & 0xf) : (byte >> 4);
    };

    const uint8_t lo_nib = src_nibble(db * 2);
    const uint8_t hi_nib = src_nibble(db * 2 + 1);
    dst_nibbles[db]      = (uint8_t) (lo_nib | (hi_nib << 4));
}

static void repack_mxfp4_soa_to_woq_scales_kernel(const uint8_t * __restrict__ e,
                                                  uint8_t * __restrict__ dst_scales,
                                                  int                      blocks_per_row,
                                                  int                      N,
                                                  int64_t                  total_entries,
                                                  const sycl::nd_item<3> & item) {
    const int64_t idx = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (idx >= total_entries) {
        return;
    }
    const int64_t kg = idx / N;
    const int64_t n  = idx - kg * N;
    dst_scales[idx]  = e[n * blocks_per_row + kg];
}

void repack_mxfp4_soa_to_woq(const void *    src,
                             uint8_t *       dst_nibbles,
                             uint8_t *       dst_scales,
                             int             blocks_per_row,
                             int             nrows,
                             dpct::queue_ptr stream) {
    const int64_t   nblocks = static_cast<int64_t>(nrows) * blocks_per_row;
    const uint8_t * qs      = static_cast<const uint8_t *>(src);
    const uint8_t * e       = qs + nblocks * (QK_MXFP4 / 2);

    const int64_t K = static_cast<int64_t>(blocks_per_row) * QK_MXFP4;
    const int64_t N = nrows;

    constexpr int WG_SIZE = 256;

    const int64_t total_nibble_bytes = (K * N) / 2;
    const int64_t n_wgs_nibbles      = (total_nibble_bytes + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs_nibbles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_soa_to_woq_nibbles_kernel(qs, dst_nibbles, blocks_per_row, nrows, total_nibble_bytes, item);
        });

    const int64_t total_scale_entries = nblocks;  // blocks_per_row * N
    const int64_t n_wgs_scales        = (total_scale_entries + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs_scales * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_soa_to_woq_scales_kernel(e, dst_scales, blocks_per_row, nrows,
                                                                   total_scale_entries, item);
                         });
}

// Fixed-capacity table of per-slot source pointers for the batched repack
// entry points below (llama.cpp-1lon). Captured BY VALUE into a SYCL kernel
// lambda -- an array of device-copyable pointers embedded in the kernel's
// argument block, the same mechanism that already carries a single pointer
// per launch in the per-slot kernels above, just widened to carry
// n_slots (<= GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS) of them. This is not a
// new device allocation or an H2D buffer upload: nothing beyond n_slots is
// ever read (the launch's expert-slot grid dimension is sized to n_slots,
// not to the table's capacity), and the table itself lives only as kernel
// argument bytes for the lifetime of the launch.
struct mxfp4_woq_repack_src_table {
    const void * ptr[GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS];
};

// Batched SOA MXFP4 -> WOQ repack kernels: identical per-slot index math to
// repack_mxfp4_soa_to_woq_{nibbles,scales}_kernel above, with an added
// expert-slot grid dimension (item.get_group(1)) so one launch covers every
// active slot. See repack_mxfp4_soa_to_woq_batched below for the shared
// per-tensor-dispatch parameters this relies on.
static void repack_mxfp4_soa_to_woq_nibbles_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                           uint8_t * __restrict__ dst_weight_base,
                                                           size_t                   weight_slot_bytes,
                                                           int                      blocks_per_row,
                                                           int                      N,
                                                           int64_t                  total_dst_bytes,
                                                           const sycl::nd_item<3> & item) {
    const int64_t db = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (db >= total_dst_bytes) {
        return;
    }
    const int       slot               = static_cast<int>(item.get_group(1));
    const uint8_t * qs                 = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_nibbles = dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes;

    auto src_nibble = [=](int64_t el) -> uint8_t {
        const int64_t k       = el / N;
        const int64_t n       = el - k * N;
        const int64_t b       = k / QK_MXFP4;
        const int64_t k_local = k - b * QK_MXFP4;
        const int64_t block_i = n * blocks_per_row + b;
        const bool    lo      = k_local < QK_MXFP4 / 2;
        const uint8_t byte    = qs[block_i * (QK_MXFP4 / 2) + (lo ? k_local : k_local - QK_MXFP4 / 2)];
        return lo ? (byte & 0xf) : (byte >> 4);
    };

    const uint8_t lo_nib = src_nibble(db * 2);
    const uint8_t hi_nib = src_nibble(db * 2 + 1);
    dst_nibbles[db]      = (uint8_t) (lo_nib | (hi_nib << 4));
}

static void repack_mxfp4_soa_to_woq_scales_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                          uint8_t * __restrict__ dst_weight_base,
                                                          size_t                   weight_slot_bytes,
                                                          size_t                   woq_nibble_slot_bytes,
                                                          int                      blocks_per_row,
                                                          int                      N,
                                                          int64_t                  nblocks,
                                                          int64_t                  total_entries,
                                                          const sycl::nd_item<3> & item) {
    const int64_t idx = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (idx >= total_entries) {
        return;
    }
    const int       slot = static_cast<int>(item.get_group(1));
    const uint8_t * qs   = static_cast<const uint8_t *>(srcs.ptr[slot]);
    const uint8_t * e    = qs + nblocks * (QK_MXFP4 / 2);
    uint8_t * __restrict__ dst_scales =
        dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes + woq_nibble_slot_bytes;

    const int64_t kg = idx / N;
    const int64_t n  = idx - kg * N;
    dst_scales[idx]  = e[n * blocks_per_row + kg];
}

void repack_mxfp4_soa_to_woq_batched(const void * const * srcs,
                                     int                  n_slots,
                                     uint8_t *            dst_weight_base,
                                     size_t               weight_slot_bytes,
                                     size_t               woq_nibble_slot_bytes,
                                     int                  blocks_per_row,
                                     int                  nrows,
                                     dpct::queue_ptr      stream) {
    GGML_ASSERT(n_slots > 0 && n_slots <= GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS);

    mxfp4_woq_repack_src_table src_table{};
    for (int i = 0; i < n_slots; ++i) {
        src_table.ptr[i] = srcs[i];
    }

    const int64_t K       = static_cast<int64_t>(blocks_per_row) * QK_MXFP4;
    const int64_t N       = nrows;
    const int64_t nblocks = static_cast<int64_t>(nrows) * blocks_per_row;

    constexpr int WG_SIZE = 256;

    const int64_t total_nibble_bytes = (K * N) / 2;
    const int64_t n_wgs_nibbles      = (total_nibble_bytes + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, n_slots, n_wgs_nibbles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_soa_to_woq_nibbles_batched_kernel(src_table, dst_weight_base, weight_slot_bytes,
                                                           blocks_per_row, static_cast<int>(N), total_nibble_bytes,
                                                           item);
        });

    const int64_t total_scale_entries = nblocks;  // blocks_per_row * N
    const int64_t n_wgs_scales        = (total_scale_entries + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, n_slots, n_wgs_scales * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_soa_to_woq_scales_batched_kernel(src_table, dst_weight_base, weight_slot_bytes,
                                                          woq_nibble_slot_bytes, blocks_per_row, static_cast<int>(N),
                                                          nblocks, total_scale_entries, item);
        });
}

// Coalesced SOA MXFP4 -> WOQ repack (perf-recovery epic, track D,
// llama.cpp-0vqt): repack_mxfp4_soa_to_woq[_batched] above are
// byte-granularity transposes -- for a fixed k (dst row), adjacent-n reads
// jump blocks_per_row*(QK_MXFP4/2) bytes apart (1440B for GPT-OSS
// blocks_per_row=90), because the SOURCE is n-major (each expert row's
// blocks_per_row*16 bytes are contiguous) while the DESTINATION wants
// k-major. Measured effective bandwidth ~16-37 GB/s on ~450 GB/s cards
// (task llama.cpp-0vqt comment c-ajw4). This SLM-tiled form reads in the
// source's native direction (coalesced within a subgroup: fixed n,
// contiguous byte-within-block j) and writes in the destination's native
// direction (coalesced: fixed k, contiguous n), bridging the two through
// workgroup-local memory -- the standard tiled-transpose pattern. Local
// memory is obtained via sycl::local_accessor + get_pointer(), the only SLM
// idiom used anywhere in this backend (e.g. dequantize_row_q4_K_sycl
// above).
//
// Nibbles grid: one workgroup per (source block b, n-tile of
// MXFP4_WOQ_REPACK_COALESCED_NBLK consecutive n's). WG_SIZE = NBLK*16 = 256
// threads, reused for both phases (reinterpreted between them, separated by
// a barrier):
//   Phase 1 (load+deinterleave): thread (n_local, j) [j fastest -- 16 lanes
//     at fixed n_local read 16 CONTIGUOUS source bytes] reads
//     qs[block_i*16+j], splits its low/high nibble (element j / j+16) into
//     local[j][n_local] / local[j+16][n_local].
//   Phase 2 (pack+store): thread (k_local, n_pair) [n_pair fastest -- lanes
//     at fixed k_local write NBLK/2 CONTIGUOUS dst bytes] packs
//     local[k_local][2*n_pair] | (local[k_local][2*n_pair+1]<<4) into
//     dst_nibbles[k*(N/2) + n0/2 + n_pair] -- exactly repack_mxfp4_soa_to_
//     woq_nibbles_kernel's db formula for N even (db=(k*N+n)/2, and k*N is
//     always even when N is even, so db=k*(N/2)+n/2 exactly).
//
// Requires N % NBLK == 0 (implies N even, needed for the db formula above
// to match the original everywhere, and guarantees every n-tile is FULL so
// no boundary masking is needed inside the nibbles kernel at all). GPT-OSS
// 20B satisfies this (N=2880, 2880/16=180 exact). The host entry point
// below checks this and falls back to the untiled kernel byte-for-byte
// otherwise -- correctness holds for any shape, the speedup is scoped to
// N%16==0.
//
// Scales grid: tiled over BOTH the k-group (b) and n axes (TILE_B x
// TILE_N, both NBLK) since there is no intra-block byte-within-block
// dimension to fold coalescing into here -- e[n*blocks_per_row+b] is
// contiguous in b for fixed n, so Phase 1 fixes n and varies b across the
// fast lane dimension; dst_scales[b*N+n] is contiguous in n for fixed b, so
// Phase 2 fixes b and varies n. blocks_per_row need not be a multiple of
// TILE_B (GPT-OSS's 90 isn't) -- the last b-tile is masked per-thread
// rather than requiring a host-side fallback; nothing about this kernel's
// correctness depends on blocks_per_row's divisibility, only nrows' (the
// same N%NBLK==0 gate as nibbles, checked once for both planes).
constexpr int MXFP4_WOQ_REPACK_COALESCED_NBLK = 16;

// SLM row stride, padded by one element (perf-recovery epic, track D
// iteration 2, llama.cpp-0vqt): a phase-1 subgroup at fixed n_local writes
// local[j*NBLK+n_local] for j=0..15 -- consecutive lanes' addresses are
// exactly NBLK bytes apart, which aliases every lane onto the SAME SLM bank
// on any hardware whose bank count divides NBLK evenly (16 and 32 both
// do), serializing what should be a single-cycle store. Padding the row to
// NBLK+1 makes the stride coprime with those bank counts, eliminating the
// systematic collision. Used by both the nibbles kernel's [32][ROW] layout
// and the scales kernel's [TILE][ROW] layout (TILE == NBLK here).
constexpr int MXFP4_WOQ_REPACK_COALESCED_ROW = MXFP4_WOQ_REPACK_COALESCED_NBLK + 1;

// Phase 1 reads the source in ONE lane-per-n_local, four-aligned-uint32
// vector loads instead of 16 lanes each doing a single scalar byte load:
// block_i*(QK_MXFP4/2) is always 16-byte aligned (block_i*16), so each of
// the four uint32 words is naturally 4-byte aligned. This cuts the memory
// instruction count 16x-per-block and lets one lane pull the whole block in
// a handful of wide transactions instead of leaning on cross-lane
// coalescing of single-byte loads.
static void repack_mxfp4_soa_to_woq_coalesced_nibbles_kernel(const uint8_t * __restrict__ qs,
                                                             uint8_t * __restrict__ dst_nibbles,
                                                             int     blocks_per_row,
                                                             int64_t N,
                                                             uint8_t * __restrict__ local,
                                                             const sycl::nd_item<3> & item) {
    constexpr int NBLK = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    constexpr int ROW  = MXFP4_WOQ_REPACK_COALESCED_ROW;
    const int     b    = static_cast<int>(item.get_group(1));
    const int64_t n0   = static_cast<int64_t>(item.get_group(2)) * NBLK;
    const int     lid  = static_cast<int>(item.get_local_id(2));

    if (lid < NBLK) {
        const int              n_local = lid;
        const int64_t          block_i = (n0 + n_local) * blocks_per_row + b;
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(qs + block_i * (QK_MXFP4 / 2));
        const uint32_t         w[4]    = { blk32[0], blk32[1], blk32[2], blk32[3] };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte              = static_cast<uint8_t>(w[j >> 2] >> ((j & 3) * 8));
            local[j * ROW + n_local]        = byte & 0xf;
            local[(j + 16) * ROW + n_local] = byte >> 4;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, perf-recovery epic, track D iteration 3,
    // llama.cpp-0vqt): the microbench isolated the 1-byte-per-lane global
    // store above as the actual bottleneck -- not the strided read
    // iteration 2 fixed -- costing ~80% of the kernel's device time on both
    // cards; a diag-write-only-vectorized reference form confirmed a
    // uint64-per-lane store recovers 4.8-5.7x. ONE lane per k_local
    // (NBLK/2==8 lanes' worth of work folded into each) assembles the WHOLE
    // row of NBLK/2 packed nibble-pair bytes in registers and issues a
    // SINGLE uint64 store, instead of NBLK/2 lanes each doing one scalar
    // byte store. Alignment: k*(N/2) and n0/2 are both always multiples of
    // 8 given the N%16==0 fast-path gate (N/2 is then a multiple of 8, and
    // n0/2=n_tile*(NBLK/2)=n_tile*8), so the store offset from dst_nibbles
    // is always 8-byte aligned; the base pointer itself is assumed at
    // least 8-byte aligned (true of every caller today -- device/cache
    // allocations are never sub-8-byte aligned). Requires NBLK/2 == 8 for
    // one uint64 to cover exactly one row (true for the committed NBLK=16).
    if (lid < 32) {
        const int     k_local = lid;
        const int64_t k       = static_cast<int64_t>(b) * QK_MXFP4 + k_local;
        uint64_t      word    = 0;
#pragma unroll
        for (int n_pair = 0; n_pair < NBLK / 2; ++n_pair) {
            const uint8_t lo = local[k_local * ROW + 2 * n_pair];
            const uint8_t hi = local[k_local * ROW + 2 * n_pair + 1];
            word |= static_cast<uint64_t>(static_cast<uint8_t>(lo | (hi << 4))) << (n_pair * 8);
        }
        uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_nibbles + k * (N / 2) + n0 / 2);
        *dst64                 = word;
    }
}

static void repack_mxfp4_soa_to_woq_coalesced_scales_kernel(const uint8_t * __restrict__ e,
                                                            uint8_t * __restrict__ dst_scales,
                                                            int     blocks_per_row,
                                                            int64_t N,
                                                            uint8_t * __restrict__ local,
                                                            const sycl::nd_item<3> & item) {
    constexpr int TILE = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    constexpr int ROW  = MXFP4_WOQ_REPACK_COALESCED_ROW;
    const int     b0   = static_cast<int>(item.get_group(1)) * TILE;
    const int64_t n0   = static_cast<int64_t>(item.get_group(2)) * TILE;
    const int     lid  = static_cast<int>(item.get_local_id(2));

    {
        const int n_local = lid / TILE;
        const int b_local = lid % TILE;
        const int b       = b0 + b_local;
        if (b < blocks_per_row) {
            local[b_local * ROW + n_local] = e[(n0 + n_local) * blocks_per_row + b];
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, iteration 3, llama.cpp-0vqt -- same fix as
    // the nibbles kernel above, applied here since the pattern is
    // identical: many lanes each doing one scalar byte store): ONE lane
    // per b_local writes its whole TILE-byte row as two uint64 stores
    // instead of TILE lanes each doing one scalar byte store. Requires
    // TILE % 8 == 0 (true for the committed TILE=16). Alignment: b*N and
    // n0 are both always multiples of 8 given the N%16==0 fast-path gate
    // (N itself a multiple of 8) and n0=n_tile*TILE=n_tile*16, so the row
    // base from dst_scales is always 8-byte aligned; same base-pointer
    // assumption as the nibbles kernel.
    if (lid < TILE) {
        const int b_local = lid;
        const int b       = b0 + b_local;
        if (b < blocks_per_row) {
            uint64_t lo_word = 0;
            uint64_t hi_word = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                lo_word |= static_cast<uint64_t>(local[b_local * ROW + i]) << (i * 8);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hi_word |= static_cast<uint64_t>(local[b_local * ROW + 8 + i]) << (i * 8);
            }
            uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_scales + static_cast<int64_t>(b) * N + n0);
            dst64[0]               = lo_word;
            dst64[1]               = hi_word;
        }
    }
}

// Host entry point: falls back to repack_mxfp4_soa_to_woq byte-for-byte
// when N % NBLK != 0 (see the file comment above) -- correctness holds for
// any (blocks_per_row, nrows), the coalesced speedup is scoped to N%16==0.
void repack_mxfp4_soa_to_woq_coalesced(const void *    src,
                                       uint8_t *       dst_nibbles,
                                       uint8_t *       dst_scales,
                                       int             blocks_per_row,
                                       int             nrows,
                                       dpct::queue_ptr stream) {
    constexpr int NBLK = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    if (nrows % NBLK != 0) {
        repack_mxfp4_soa_to_woq(src, dst_nibbles, dst_scales, blocks_per_row, nrows, stream);
        return;
    }

    // Quality-review R4 (llama.cpp-0vqt): the nibbles/scales kernels'
    // vectorized uint64 stores (iteration 3) require both destination
    // bases to be 8-byte aligned -- a misaligned base is UB that would
    // silently corrupt adjacent bytes rather than fault, not something the
    // prose-only alignment argument in the kernel comments can catch on
    // its own. Production callers satisfy this via align_up_256 slot
    // strides (ggml-sycl.cpp); this is the tripwire for any future caller
    // that doesn't.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_nibbles) % 8 == 0);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_scales) % 8 == 0);

    const int64_t   nblocks = static_cast<int64_t>(nrows) * blocks_per_row;
    const uint8_t * qs      = static_cast<const uint8_t *>(src);
    const uint8_t * e       = qs + nblocks * (QK_MXFP4 / 2);
    const int64_t   N       = nrows;

    // Nibbles WG_SIZE dropped 256 -> 32 in iteration 3 (llama.cpp-0vqt):
    // phase 1 only ever used NBLK(=16) active lanes and phase 2 (after the
    // write-vectorization above) only ever uses 32 -- 256 was oversized for
    // both phases post-iteration-2, launching mostly-idle lanes. Scales
    // keeps WG_SIZE=256 -- its phase 1 genuinely uses all 256 lanes (a
    // TILE x TILE 2D tile, unchanged by this iteration).
    constexpr int WG_SIZE_NIBBLES = 32;
    constexpr int WG_SIZE         = 256;
    const int64_t n_tiles         = N / NBLK;

    constexpr int ROW = MXFP4_WOQ_REPACK_COALESCED_ROW;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(32 * ROW), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, n_tiles * WG_SIZE_NIBBLES),
                                           sycl::range<3>(1, 1, WG_SIZE_NIBBLES)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_soa_to_woq_coalesced_nibbles_kernel(qs, dst_nibbles, blocks_per_row, N,
                                                                              get_pointer(local_acc), item);
                         });
    });

    const int64_t b_tiles = (blocks_per_row + NBLK - 1) / NBLK;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(NBLK * ROW), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, b_tiles, n_tiles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
            [=](sycl::nd_item<3> item) {
                repack_mxfp4_soa_to_woq_coalesced_scales_kernel(e, dst_scales, blocks_per_row, N,
                                                                get_pointer(local_acc), item);
            });
    });
}

static void repack_mxfp4_soa_to_woq_coalesced_nibbles_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                     uint8_t * __restrict__ dst_weight_base,
                                                                     size_t  weight_slot_bytes,
                                                                     int     blocks_per_row,
                                                                     int64_t N,
                                                                     int64_t n_tiles,
                                                                     uint8_t * __restrict__ local,
                                                                     const sycl::nd_item<3> & item) {
    constexpr int NBLK     = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    constexpr int ROW      = MXFP4_WOQ_REPACK_COALESCED_ROW;
    const int     slot     = static_cast<int>(item.get_group(1));
    const int64_t tile_lin = static_cast<int64_t>(item.get_group(2));
    const int     b        = static_cast<int>(tile_lin / n_tiles);
    const int64_t n0       = (tile_lin % n_tiles) * NBLK;
    const int     lid      = static_cast<int>(item.get_local_id(2));

    const uint8_t * qs                 = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_nibbles = dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes;

    if (lid < NBLK) {
        const int              n_local = lid;
        const int64_t          block_i = (n0 + n_local) * blocks_per_row + b;
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(qs + block_i * (QK_MXFP4 / 2));
        const uint32_t         w[4]    = { blk32[0], blk32[1], blk32[2], blk32[3] };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte              = static_cast<uint8_t>(w[j >> 2] >> ((j & 3) * 8));
            local[j * ROW + n_local]        = byte & 0xf;
            local[(j + 16) * ROW + n_local] = byte >> 4;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, iteration 3) -- see
    // repack_mxfp4_soa_to_woq_coalesced_nibbles_kernel above for the
    // derivation; identical here.
    if (lid < 32) {
        const int     k_local = lid;
        const int64_t k       = static_cast<int64_t>(b) * QK_MXFP4 + k_local;
        uint64_t      word    = 0;
#pragma unroll
        for (int n_pair = 0; n_pair < NBLK / 2; ++n_pair) {
            const uint8_t lo = local[k_local * ROW + 2 * n_pair];
            const uint8_t hi = local[k_local * ROW + 2 * n_pair + 1];
            word |= static_cast<uint64_t>(static_cast<uint8_t>(lo | (hi << 4))) << (n_pair * 8);
        }
        uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_nibbles + k * (N / 2) + n0 / 2);
        *dst64                 = word;
    }
}

static void repack_mxfp4_soa_to_woq_coalesced_scales_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                    uint8_t * __restrict__ dst_weight_base,
                                                                    size_t  weight_slot_bytes,
                                                                    size_t  woq_nibble_slot_bytes,
                                                                    int     blocks_per_row,
                                                                    int64_t N,
                                                                    int64_t nblocks,
                                                                    int64_t n_tiles,
                                                                    uint8_t * __restrict__ local,
                                                                    const sycl::nd_item<3> & item) {
    constexpr int TILE     = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    constexpr int ROW      = MXFP4_WOQ_REPACK_COALESCED_ROW;
    const int     slot     = static_cast<int>(item.get_group(1));
    const int64_t tile_lin = static_cast<int64_t>(item.get_group(2));
    const int     b0       = static_cast<int>(tile_lin / n_tiles) * TILE;
    const int64_t n0       = (tile_lin % n_tiles) * TILE;
    const int     lid      = static_cast<int>(item.get_local_id(2));

    const uint8_t * qs = static_cast<const uint8_t *>(srcs.ptr[slot]);
    const uint8_t * e  = qs + nblocks * (QK_MXFP4 / 2);
    uint8_t * __restrict__ dst_scales =
        dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes + woq_nibble_slot_bytes;

    {
        const int n_local = lid / TILE;
        const int b_local = lid % TILE;
        const int b       = b0 + b_local;
        if (b < blocks_per_row) {
            local[b_local * ROW + n_local] = e[(n0 + n_local) * blocks_per_row + b];
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, iteration 3) -- see
    // repack_mxfp4_soa_to_woq_coalesced_scales_kernel above for the
    // derivation; identical here.
    if (lid < TILE) {
        const int b_local = lid;
        const int b       = b0 + b_local;
        if (b < blocks_per_row) {
            uint64_t lo_word = 0;
            uint64_t hi_word = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                lo_word |= static_cast<uint64_t>(local[b_local * ROW + i]) << (i * 8);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hi_word |= static_cast<uint64_t>(local[b_local * ROW + 8 + i]) << (i * 8);
            }
            uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_scales + static_cast<int64_t>(b) * N + n0);
            dst64[0]               = lo_word;
            dst64[1]               = hi_word;
        }
    }
}

// Batched form of repack_mxfp4_soa_to_woq_coalesced above: adds the same
// expert-slot grid dimension as repack_mxfp4_soa_to_woq_batched, folded
// into dim2 alongside the (b, n-tile) linearization (dim1 carries the slot
// index, matching the existing *_batched kernels' convention). Falls back
// to repack_mxfp4_soa_to_woq_batched byte-for-byte on the same N%NBLK!=0
// condition as the per-slot form above.
void repack_mxfp4_soa_to_woq_coalesced_batched(const void * const * srcs,
                                               int                  n_slots,
                                               uint8_t *            dst_weight_base,
                                               size_t               weight_slot_bytes,
                                               size_t               woq_nibble_slot_bytes,
                                               int                  blocks_per_row,
                                               int                  nrows,
                                               dpct::queue_ptr      stream) {
    GGML_ASSERT(n_slots > 0 && n_slots <= GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS);

    constexpr int NBLK = MXFP4_WOQ_REPACK_COALESCED_NBLK;
    if (nrows % NBLK != 0) {
        repack_mxfp4_soa_to_woq_batched(srcs, n_slots, dst_weight_base, weight_slot_bytes, woq_nibble_slot_bytes,
                                        blocks_per_row, nrows, stream);
        return;
    }

    // Quality-review R4 (llama.cpp-0vqt): the vectorized uint64 stores need
    // the batch base AND every per-slot store address 8-byte aligned --
    // the latter holds for any slot iff weight_slot_bytes (nibbles' slot
    // stride) and woq_nibble_slot_bytes (the scales offset within a slot)
    // are themselves 8-byte aligned, since slot*weight_slot_bytes and
    // +woq_nibble_slot_bytes are then multiples of 8 regardless of slot.
    // See the per-slot form above for why a misaligned base is
    // silent-corruption UB, not a fault.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_weight_base) % 8 == 0);
    GGML_ASSERT(weight_slot_bytes % 8 == 0);
    GGML_ASSERT(woq_nibble_slot_bytes % 8 == 0);

    mxfp4_woq_repack_src_table src_table{};
    for (int i = 0; i < n_slots; ++i) {
        src_table.ptr[i] = srcs[i];
    }

    const int64_t N               = nrows;
    const int64_t nblocks         = static_cast<int64_t>(nrows) * blocks_per_row;
    // Nibbles WG_SIZE dropped 256 -> 32 in iteration 3 -- see
    // repack_mxfp4_soa_to_woq_coalesced above for why; scales keeps 256.
    constexpr int WG_SIZE_NIBBLES = 32;
    constexpr int WG_SIZE         = 256;
    const int64_t n_tiles         = N / NBLK;

    constexpr int ROW = MXFP4_WOQ_REPACK_COALESCED_ROW;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(32 * ROW), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, n_slots, blocks_per_row * n_tiles * WG_SIZE_NIBBLES),
                                           sycl::range<3>(1, 1, WG_SIZE_NIBBLES)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_soa_to_woq_coalesced_nibbles_batched_kernel(
                                 src_table, dst_weight_base, weight_slot_bytes, blocks_per_row, N, n_tiles,
                                 get_pointer(local_acc), item);
                         });
    });

    const int64_t b_tiles = (blocks_per_row + NBLK - 1) / NBLK;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(NBLK * ROW), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, n_slots, b_tiles * n_tiles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
            [=](sycl::nd_item<3> item) {
                repack_mxfp4_soa_to_woq_coalesced_scales_batched_kernel(src_table, dst_weight_base, weight_slot_bytes,
                                                                        woq_nibble_slot_bytes, blocks_per_row, N,
                                                                        nblocks, n_tiles, get_pointer(local_acc), item);
            });
    });
}

// XMX_TILED MXFP4 -> WOQ repack: inverts the tiled materializer's byte
// layout into the SAME destination WOQ layout as repack_mxfp4_soa_to_woq
// above (perf-recovery epic, track C2b Option T, llama.cpp-ntfx; nibble
// order + scale layout confirmed by the C1 spike, llama.cpp-4m9p comment
// c-a5by): sequential nibble order for weights {K,N} strides {N,1}, e8m0
// scales {K/QK_MXFP4,N} strides {N,1}.
//
// Source (XMX_TILED, one weight matrix / expert-slot; k-tile-major
// [tile_k_group][tile_n_group] -- moe-xmx-fused.hpp:118-153, written by
// ggml_sycl::moe_tile_convert::reorder_mxfp4_to_xmx_tiled /
// reorder_mxfp4_aos_to_xmx_tiled, moe-tile-convert.cpp:24-173, both of which
// produce byte-identical output): the tile-group grid has n_tile_groups_k =
// blocks_per_row groups along K (moe-xmx-fused.hpp:136-141:
// tiles_k_per_group is always 1, i.e. one QK_MXFP4=32-element K-block per
// group, so blocks_per_row IS the K-group count -- this function takes it
// directly, the same convention as repack_mxfp4_soa_to_woq's parameter) and
// n_tile_groups_n = ceil(nrows / tile_n_total) groups along N. Each tile
// group is tile_n_total * (1 + QK_MXFP4/2) bytes (moe-tile-convert.cpp:52,
// 131): tile_n_total scale bytes (one per output row, in row order) followed
// by tile_n_total 16-byte quantized blocks (one per row, same order --
// moe-tile-convert.cpp:70-99 verbatim-copies each row's source 16-byte
// block into its slot, so the intra-block nibble packing is the SAME
// j/j+16 interleave that dequantize_tile_mxfp4_soa_rowmajor above decodes:
// byte j (j=0..15) holds element j in its low nibble, element j+16 in its
// high nibble, for the block's QK_MXFP4=32 elements).
//
// Inverse map for destination element (k, n), k in [0,K), n in [0,N):
//   k_block = k / QK_MXFP4;  k_local = k - k_block * QK_MXFP4
//   tg_n = n / tile_n_total; tn = n - tg_n * tile_n_total
//   n_tile_groups_n = ceil(N / tile_n_total)
//   group_bytes  = tile_n_total * (1 + QK_MXFP4/2)
//   group_offset = (k_block * n_tile_groups_n + tg_n) * group_bytes
//   scale byte   = src_tiled[group_offset + tn]
//   qs byte      = src_tiled[group_offset + tile_n_total + tn*(QK_MXFP4/2)
//                             + (k_local < QK_MXFP4/2 ? k_local : k_local - QK_MXFP4/2)]
//   nibble       = k_local < QK_MXFP4/2 ? (qs byte & 0xf) : (qs byte >> 4)
//
// Caps assumed by the planned call site (batched WOQ PP executor, task C3 --
// not yet landed; the device test is the only caller today) (device caps
// log): caps.K=32, caps.N=16, GGML_SYCL_MXFP4_MOE_XMX_SG=32,
// optimal_tiles_n=1 -> tile_n_total=16. This function does NOT hardcode
// those numbers -- blocks_per_row, nrows and tile_n_total are taken as
// arguments derived from device caps at the call site, since caps can
// differ across devices/builds.
static void repack_mxfp4_xmx_tiled_to_woq_nibbles_kernel(const uint8_t * __restrict__ src_tiled,
                                                         uint8_t * __restrict__ dst_nibbles,
                                                         int                      N,
                                                         int                      tile_n_total,
                                                         int64_t                  n_tile_groups_n,
                                                         int64_t                  group_bytes,
                                                         int64_t                  total_dst_bytes,
                                                         const sycl::nd_item<3> & item) {
    const int64_t db = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (db >= total_dst_bytes) {
        return;
    }

    auto src_nibble = [=](int64_t el) -> uint8_t {
        const int64_t k            = el / N;
        const int64_t n            = el - k * N;
        const int64_t k_block      = k / QK_MXFP4;
        const int64_t k_local      = k - k_block * QK_MXFP4;
        const int64_t tg_n         = n / tile_n_total;
        const int64_t tn           = n - tg_n * tile_n_total;
        const int64_t group_offset = (k_block * n_tile_groups_n + tg_n) * group_bytes;
        const bool    lo           = k_local < QK_MXFP4 / 2;
        const uint8_t byte         = src_tiled[group_offset + tile_n_total + tn * (QK_MXFP4 / 2) +
                                        (lo ? k_local : k_local - QK_MXFP4 / 2)];
        return lo ? (byte & 0xf) : (byte >> 4);
    };

    const uint8_t lo_nib = src_nibble(db * 2);
    const uint8_t hi_nib = src_nibble(db * 2 + 1);
    dst_nibbles[db]      = (uint8_t) (lo_nib | (hi_nib << 4));
}

static void repack_mxfp4_xmx_tiled_to_woq_scales_kernel(const uint8_t * __restrict__ src_tiled,
                                                        uint8_t * __restrict__ dst_scales,
                                                        int                      N,
                                                        int                      tile_n_total,
                                                        int64_t                  n_tile_groups_n,
                                                        int64_t                  group_bytes,
                                                        int64_t                  total_entries,
                                                        const sycl::nd_item<3> & item) {
    const int64_t idx = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (idx >= total_entries) {
        return;
    }
    const int64_t k_block      = idx / N;
    const int64_t n            = idx - k_block * N;
    const int64_t tg_n         = n / tile_n_total;
    const int64_t tn           = n - tg_n * tile_n_total;
    const int64_t group_offset = (k_block * n_tile_groups_n + tg_n) * group_bytes;
    dst_scales[idx]            = src_tiled[group_offset + tn];
}

void repack_mxfp4_xmx_tiled_to_woq(const void *    src_tiled,
                                   uint8_t *       dst_nibbles,
                                   uint8_t *       dst_scales,
                                   int             blocks_per_row,
                                   int             nrows,
                                   int             tile_n_total,
                                   dpct::queue_ptr stream) {
    const uint8_t * src = static_cast<const uint8_t *>(src_tiled);

    const int64_t K = static_cast<int64_t>(blocks_per_row) * QK_MXFP4;
    const int64_t N = nrows;

    const int64_t n_tile_groups_n = (N + tile_n_total - 1) / tile_n_total;
    const int64_t group_bytes     = static_cast<int64_t>(tile_n_total) * (1 + QK_MXFP4 / 2);

    constexpr int WG_SIZE = 256;

    const int64_t total_nibble_bytes = (K * N) / 2;
    const int64_t n_wgs_nibbles      = (total_nibble_bytes + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs_nibbles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_xmx_tiled_to_woq_nibbles_kernel(src, dst_nibbles, nrows, tile_n_total, n_tile_groups_n,
                                                         group_bytes, total_nibble_bytes, item);
        });

    const int64_t total_scale_entries = static_cast<int64_t>(blocks_per_row) * N;
    const int64_t n_wgs_scales        = (total_scale_entries + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_wgs_scales * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_xmx_tiled_to_woq_scales_kernel(src, dst_scales, nrows, tile_n_total, n_tile_groups_n,
                                                        group_bytes, total_scale_entries, item);
        });
}

// Batched XMX_TILED MXFP4 -> WOQ repack kernels: identical per-slot index
// math to repack_mxfp4_xmx_tiled_to_woq_{nibbles,scales}_kernel above, with
// an added expert-slot grid dimension (item.get_group(1)) so one launch
// covers every active slot -- see repack_mxfp4_soa_to_woq_nibbles_batched_
// kernel above for the shared source-table mechanism.
static void repack_mxfp4_xmx_tiled_to_woq_nibbles_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                 uint8_t * __restrict__ dst_weight_base,
                                                                 size_t                   weight_slot_bytes,
                                                                 int                      N,
                                                                 int                      tile_n_total,
                                                                 int64_t                  n_tile_groups_n,
                                                                 int64_t                  group_bytes,
                                                                 int64_t                  total_dst_bytes,
                                                                 const sycl::nd_item<3> & item) {
    const int64_t db = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (db >= total_dst_bytes) {
        return;
    }
    const int       slot               = static_cast<int>(item.get_group(1));
    const uint8_t * src_tiled          = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_nibbles = dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes;

    auto src_nibble = [=](int64_t el) -> uint8_t {
        const int64_t k            = el / N;
        const int64_t n            = el - k * N;
        const int64_t k_block      = k / QK_MXFP4;
        const int64_t k_local      = k - k_block * QK_MXFP4;
        const int64_t tg_n         = n / tile_n_total;
        const int64_t tn           = n - tg_n * tile_n_total;
        const int64_t group_offset = (k_block * n_tile_groups_n + tg_n) * group_bytes;
        const bool    lo           = k_local < QK_MXFP4 / 2;
        const uint8_t byte =
            src_tiled[group_offset + tile_n_total + tn * (QK_MXFP4 / 2) + (lo ? k_local : k_local - QK_MXFP4 / 2)];
        return lo ? (byte & 0xf) : (byte >> 4);
    };

    const uint8_t lo_nib = src_nibble(db * 2);
    const uint8_t hi_nib = src_nibble(db * 2 + 1);
    dst_nibbles[db]      = (uint8_t) (lo_nib | (hi_nib << 4));
}

static void repack_mxfp4_xmx_tiled_to_woq_scales_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                uint8_t * __restrict__ dst_weight_base,
                                                                size_t                   weight_slot_bytes,
                                                                size_t                   woq_nibble_slot_bytes,
                                                                int                      N,
                                                                int                      tile_n_total,
                                                                int64_t                  n_tile_groups_n,
                                                                int64_t                  group_bytes,
                                                                int64_t                  total_entries,
                                                                const sycl::nd_item<3> & item) {
    const int64_t idx = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
    if (idx >= total_entries) {
        return;
    }
    const int       slot      = static_cast<int>(item.get_group(1));
    const uint8_t * src_tiled = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_scales =
        dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes + woq_nibble_slot_bytes;

    const int64_t k_block      = idx / N;
    const int64_t n            = idx - k_block * N;
    const int64_t tg_n         = n / tile_n_total;
    const int64_t tn           = n - tg_n * tile_n_total;
    const int64_t group_offset = (k_block * n_tile_groups_n + tg_n) * group_bytes;
    dst_scales[idx]            = src_tiled[group_offset + tn];
}

void repack_mxfp4_xmx_tiled_to_woq_batched(const void * const * srcs,
                                           int                  n_slots,
                                           uint8_t *            dst_weight_base,
                                           size_t               weight_slot_bytes,
                                           size_t               woq_nibble_slot_bytes,
                                           int                  blocks_per_row,
                                           int                  nrows,
                                           int                  tile_n_total,
                                           dpct::queue_ptr      stream) {
    GGML_ASSERT(n_slots > 0 && n_slots <= GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS);

    mxfp4_woq_repack_src_table src_table{};
    for (int i = 0; i < n_slots; ++i) {
        src_table.ptr[i] = srcs[i];
    }

    const int64_t K = static_cast<int64_t>(blocks_per_row) * QK_MXFP4;
    const int64_t N = nrows;

    const int64_t n_tile_groups_n = (N + tile_n_total - 1) / tile_n_total;
    const int64_t group_bytes     = static_cast<int64_t>(tile_n_total) * (1 + QK_MXFP4 / 2);

    constexpr int WG_SIZE = 256;

    const int64_t total_nibble_bytes = (K * N) / 2;
    const int64_t n_wgs_nibbles      = (total_nibble_bytes + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, n_slots, n_wgs_nibbles * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_xmx_tiled_to_woq_nibbles_batched_kernel(src_table, dst_weight_base, weight_slot_bytes,
                                                                 static_cast<int>(N), tile_n_total, n_tile_groups_n,
                                                                 group_bytes, total_nibble_bytes, item);
        });

    const int64_t total_scale_entries = static_cast<int64_t>(blocks_per_row) * N;
    const int64_t n_wgs_scales        = (total_scale_entries + WG_SIZE - 1) / WG_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, n_slots, n_wgs_scales * WG_SIZE), sycl::range<3>(1, 1, WG_SIZE)),
        [=](sycl::nd_item<3> item) {
            repack_mxfp4_xmx_tiled_to_woq_scales_batched_kernel(
                src_table, dst_weight_base, weight_slot_bytes, woq_nibble_slot_bytes, static_cast<int>(N), tile_n_total,
                n_tile_groups_n, group_bytes, total_scale_entries, item);
        });
}

// Coalesced XMX_TILED MXFP4 -> WOQ repack (perf-recovery epic, track D,
// llama.cpp-0vqt): same read/write-direction mismatch as the SOA form
// above, but the XMX_TILED source is friendlier -- within one source tile
// group (fixed k_block, fixed tg_n), all tile_n_total rows' 16-byte qs
// blocks are laid out back-to-back (moe-tile-convert.cpp), so the whole
// group's qs plane (tile_n_total*16 bytes) is ONE contiguous run rather
// than SOA's blocks_per_row*16-byte-separated rows. Choosing NBLK =
// tile_n_total makes one workgroup = one full source tile group: Phase 1
// (load+deinterleave) is the same lane mapping as the SOA kernel (n_local,
// j fastest) but the whole tile's read is one contiguous span; Phase 2
// (pack+store) is byte-identical logic to the SOA form, since the
// destination layout is the same {K,N}/{K/32,N} WOQ layout either way.
// tile_n_total is a runtime (caps-dependent) value, so the SLM extent is
// sized dynamically via the local_accessor's range argument, not a
// compile-time constant.
//
// Scales are even simpler here: the source scale bytes for a tile group
// (src_tiled[group_offset .. group_offset+tile_n_total)) and the
// destination row dst_scales[b*N+n0 .. +tile_n_total) are BOTH already
// contiguous in n -- no transpose, no SLM, a direct per-element copy.
//
// Requires tile_n_total == 16 EXACTLY (even was sufficient for the
// nibble-pair-packing math alone, and iteration 2's vectorized uint32
// phase-1 reads only needed a multiple of 4 for 4-byte alignment; iteration
// 3 tightened this further to 16 exactly, since the write phase's uint64
// store always writes a full 8-byte row and would zero-pad past a
// narrower row into the next n-tile group's bytes -- see the host entry
// point below for the derivation), and N % tile_n_total == 0 (every tile
// group full, matching the doc's caps: tile_n_total=16, N=2880 -> 180
// exact groups). Falls back to the untiled kernel byte-for-byte otherwise.
//
// Iteration 2 (llama.cpp-0vqt, same two fixes as the SOA kernel above):
// `row` pads the SLM row stride to tile_n_total+1 to avoid the phase-1
// bank-conflict stride (passed in rather than a compile-time constant,
// since tile_n_total is a runtime caps-dependent value here); phase 1
// reads one lane's whole 16-byte block via four aligned uint32 words
// instead of 16 scalar byte loads (the tile group's qs plane is one
// contiguous run, so this is even more straightforwardly aligned than the
// SOA case).
static void repack_mxfp4_xmx_tiled_to_woq_coalesced_nibbles_kernel(const uint8_t * __restrict__ src_tiled,
                                                                   uint8_t * __restrict__ dst_nibbles,
                                                                   int64_t N,
                                                                   int     tile_n_total,
                                                                   int     row,
                                                                   int64_t group_bytes,
                                                                   uint8_t * __restrict__ local,
                                                                   const sycl::nd_item<3> & item) {
    const int     b            = static_cast<int>(item.get_group(1));
    const int64_t tgn          = static_cast<int64_t>(item.get_group(2));
    const int64_t n0           = tgn * tile_n_total;
    const int     lid          = static_cast<int>(item.get_local_id(2));
    const int64_t group_offset = (static_cast<int64_t>(b) * (N / tile_n_total) + tgn) * group_bytes;

    if (lid < tile_n_total) {
        const int              n_local = lid;
        const int64_t          base    = group_offset + tile_n_total + static_cast<int64_t>(n_local) * (QK_MXFP4 / 2);
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(src_tiled + base);
        const uint32_t         w[4]    = { blk32[0], blk32[1], blk32[2], blk32[3] };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte              = static_cast<uint8_t>(w[j >> 2] >> ((j & 3) * 8));
            local[j * row + n_local]        = byte & 0xf;
            local[(j + 16) * row + n_local] = byte >> 4;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, iteration 3, llama.cpp-0vqt) -- see
    // repack_mxfp4_soa_to_woq_coalesced_nibbles_kernel for the derivation.
    // ONE lane per k_local packs the whole tile_n_total/2-byte row into a
    // single uint64 store. SAFE ONLY because the host entry point below
    // tightens the fast-path gate to tile_n_total == 16 EXACTLY (not just
    // a multiple of 4, as iteration 2 allowed): with n_pair_count == 8, the
    // store's 8 bytes exactly cover the row -- for any smaller
    // tile_n_total the same store would still write a full 8 bytes and
    // zero-pad past the row's real width into the NEXT n-tile group's
    // bytes, corrupting data another workgroup owns. Do not widen the gate
    // back to %4==0 without also bounding/masking this store.
    if (lid < 32) {
        const int     k_local      = lid;
        const int64_t k            = static_cast<int64_t>(b) * QK_MXFP4 + k_local;
        const int     n_pair_count = tile_n_total / 2;
        uint64_t      word         = 0;
        for (int n_pair = 0; n_pair < n_pair_count; ++n_pair) {
            const uint8_t lo = local[k_local * row + 2 * n_pair];
            const uint8_t hi = local[k_local * row + 2 * n_pair + 1];
            word |= static_cast<uint64_t>(static_cast<uint8_t>(lo | (hi << 4))) << (n_pair * 8);
        }
        uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_nibbles + k * (N / 2) + n0 / 2);
        *dst64                 = word;
    }
}

static void repack_mxfp4_xmx_tiled_to_woq_coalesced_scales_kernel(const uint8_t * __restrict__ src_tiled,
                                                                  uint8_t * __restrict__ dst_scales,
                                                                  int64_t                  N,
                                                                  int                      tile_n_total,
                                                                  int64_t                  group_bytes,
                                                                  const sycl::nd_item<3> & item) {
    const int     b            = static_cast<int>(item.get_group(1));
    const int64_t tgn          = static_cast<int64_t>(item.get_group(2));
    const int64_t n0           = tgn * tile_n_total;
    const int     n_local      = static_cast<int>(item.get_local_id(2));
    const int64_t group_offset = (static_cast<int64_t>(b) * (N / tile_n_total) + tgn) * group_bytes;

    dst_scales[static_cast<int64_t>(b) * N + n0 + n_local] = src_tiled[group_offset + n_local];
}

// Host entry point: falls back to repack_mxfp4_xmx_tiled_to_woq
// byte-for-byte when the fast-path preconditions don't hold (see the file
// comment above).
void repack_mxfp4_xmx_tiled_to_woq_coalesced(const void *    src_tiled,
                                             uint8_t *       dst_nibbles,
                                             uint8_t *       dst_scales,
                                             int             blocks_per_row,
                                             int             nrows,
                                             int             tile_n_total,
                                             dpct::queue_ptr stream) {
    const int64_t N = nrows;
    // Iteration 3 (llama.cpp-0vqt) tightened this from tile_n_total % 4 ==
    // 0 to tile_n_total == 16 EXACTLY: the vectorized phase-2 write below
    // packs a fixed 8-byte (tile_n_total/2) row into one uint64 store, and
    // that store is only safe when the row is EXACTLY 8 bytes wide --
    // anything smaller would zero-pad past the real row into the next
    // n-tile group's bytes (see the nibbles kernel's comment). The only
    // value ever seen in practice (device caps: tile_n_total=16) is
    // unaffected; any other value falls back to the untiled kernel.
    if (tile_n_total != 16 || N % tile_n_total != 0) {
        repack_mxfp4_xmx_tiled_to_woq(src_tiled, dst_nibbles, dst_scales, blocks_per_row, nrows, tile_n_total, stream);
        return;
    }

    // Quality-review R4 (llama.cpp-0vqt): see repack_mxfp4_soa_to_woq_
    // coalesced above for why -- same tripwire for the nibbles kernel's
    // vectorized uint64 store base.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_nibbles) % 8 == 0);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_scales) % 8 == 0);

    const uint8_t * src             = static_cast<const uint8_t *>(src_tiled);
    const int64_t   n_tile_groups_n = N / tile_n_total;
    const int64_t   group_bytes     = static_cast<int64_t>(tile_n_total) * (1 + QK_MXFP4 / 2);
    // wg_nibbles dropped from tile_n_total*16 (=256) to 32 in iteration 3 --
    // see repack_mxfp4_soa_to_woq_coalesced for why (phase 1 only ever used
    // tile_n_total(<=16) active lanes, phase 2 only ever uses 32 now).
    constexpr int   wg_nibbles      = 32;
    const int       row             = tile_n_total + 1;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(32 * row), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, n_tile_groups_n * wg_nibbles),
                                           sycl::range<3>(1, 1, wg_nibbles)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_xmx_tiled_to_woq_coalesced_nibbles_kernel(
                                 src, dst_nibbles, N, tile_n_total, row, group_bytes, get_pointer(local_acc), item);
                         });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, blocks_per_row, n_tile_groups_n * tile_n_total),
                                           sycl::range<3>(1, 1, tile_n_total)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_xmx_tiled_to_woq_coalesced_scales_kernel(src, dst_scales, N, tile_n_total,
                                                                                   group_bytes, item);
                         });
    });
}

static void repack_mxfp4_xmx_tiled_to_woq_coalesced_nibbles_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                           uint8_t * __restrict__ dst_weight_base,
                                                                           size_t  weight_slot_bytes,
                                                                           int64_t N,
                                                                           int     tile_n_total,
                                                                           int     row,
                                                                           int64_t n_tile_groups_n,
                                                                           int64_t group_bytes,
                                                                           uint8_t * __restrict__ local,
                                                                           const sycl::nd_item<3> & item) {
    const int     slot     = static_cast<int>(item.get_group(1));
    const int64_t tile_lin = static_cast<int64_t>(item.get_group(2));
    const int     b        = static_cast<int>(tile_lin / n_tile_groups_n);
    const int64_t tgn      = tile_lin % n_tile_groups_n;
    const int64_t n0       = tgn * tile_n_total;
    const int     lid      = static_cast<int>(item.get_local_id(2));

    const uint8_t * src_tiled          = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_nibbles = dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes;
    const int64_t group_offset         = (static_cast<int64_t>(b) * n_tile_groups_n + tgn) * group_bytes;

    if (lid < tile_n_total) {
        const int              n_local = lid;
        const int64_t          base    = group_offset + tile_n_total + static_cast<int64_t>(n_local) * (QK_MXFP4 / 2);
        const uint32_t * const blk32   = reinterpret_cast<const uint32_t *>(src_tiled + base);
        const uint32_t         w[4]    = { blk32[0], blk32[1], blk32[2], blk32[3] };
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte              = static_cast<uint8_t>(w[j >> 2] >> ((j & 3) * 8));
            local[j * row + n_local]        = byte & 0xf;
            local[(j + 16) * row + n_local] = byte >> 4;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    // Phase 2 (vectorized write, iteration 3) -- see
    // repack_mxfp4_xmx_tiled_to_woq_coalesced_nibbles_kernel above for the
    // derivation and the tile_n_total==16-exactly safety requirement
    // (enforced by the batched host entry point below).
    if (lid < 32) {
        const int     k_local      = lid;
        const int64_t k            = static_cast<int64_t>(b) * QK_MXFP4 + k_local;
        const int     n_pair_count = tile_n_total / 2;
        uint64_t      word         = 0;
        for (int n_pair = 0; n_pair < n_pair_count; ++n_pair) {
            const uint8_t lo = local[k_local * row + 2 * n_pair];
            const uint8_t hi = local[k_local * row + 2 * n_pair + 1];
            word |= static_cast<uint64_t>(static_cast<uint8_t>(lo | (hi << 4))) << (n_pair * 8);
        }
        uint64_t * const dst64 = reinterpret_cast<uint64_t *>(dst_nibbles + k * (N / 2) + n0 / 2);
        *dst64                 = word;
    }
}

static void repack_mxfp4_xmx_tiled_to_woq_coalesced_scales_batched_kernel(mxfp4_woq_repack_src_table srcs,
                                                                          uint8_t * __restrict__ dst_weight_base,
                                                                          size_t  weight_slot_bytes,
                                                                          size_t  woq_nibble_slot_bytes,
                                                                          int64_t N,
                                                                          int     tile_n_total,
                                                                          int64_t n_tile_groups_n,
                                                                          int64_t group_bytes,
                                                                          const sycl::nd_item<3> & item) {
    const int     slot     = static_cast<int>(item.get_group(1));
    const int64_t tile_lin = static_cast<int64_t>(item.get_group(2));
    const int     b        = static_cast<int>(tile_lin / n_tile_groups_n);
    const int64_t tgn      = tile_lin % n_tile_groups_n;
    const int64_t n0       = tgn * tile_n_total;
    const int     n_local  = static_cast<int>(item.get_local_id(2));

    const uint8_t * src_tiled = static_cast<const uint8_t *>(srcs.ptr[slot]);
    uint8_t * __restrict__ dst_scales =
        dst_weight_base + static_cast<size_t>(slot) * weight_slot_bytes + woq_nibble_slot_bytes;
    const int64_t group_offset = (static_cast<int64_t>(b) * n_tile_groups_n + tgn) * group_bytes;

    dst_scales[static_cast<int64_t>(b) * N + n0 + n_local] = src_tiled[group_offset + n_local];
}

// Batched form of repack_mxfp4_xmx_tiled_to_woq_coalesced above, same
// slot-in-dim1 convention as repack_mxfp4_soa_to_woq_coalesced_batched.
// Falls back to repack_mxfp4_xmx_tiled_to_woq_batched byte-for-byte on the
// same preconditions as the per-slot form above.
void repack_mxfp4_xmx_tiled_to_woq_coalesced_batched(const void * const * srcs,
                                                     int                  n_slots,
                                                     uint8_t *            dst_weight_base,
                                                     size_t               weight_slot_bytes,
                                                     size_t               woq_nibble_slot_bytes,
                                                     int                  blocks_per_row,
                                                     int                  nrows,
                                                     int                  tile_n_total,
                                                     dpct::queue_ptr      stream) {
    GGML_ASSERT(n_slots > 0 && n_slots <= GGML_SYCL_MXFP4_WOQ_REPACK_MAX_SLOTS);

    const int64_t N = nrows;
    // See repack_mxfp4_xmx_tiled_to_woq_coalesced above for why tile_n_total
    // must equal 16 EXACTLY (not just be a multiple of 4) as of iteration 3.
    if (tile_n_total != 16 || N % tile_n_total != 0) {
        repack_mxfp4_xmx_tiled_to_woq_batched(srcs, n_slots, dst_weight_base, weight_slot_bytes, woq_nibble_slot_bytes,
                                              blocks_per_row, nrows, tile_n_total, stream);
        return;
    }

    // Quality-review R4 (llama.cpp-0vqt): see repack_mxfp4_soa_to_woq_
    // coalesced_batched above for why -- same tripwire for the batch base
    // and the per-slot store strides.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst_weight_base) % 8 == 0);
    GGML_ASSERT(weight_slot_bytes % 8 == 0);
    GGML_ASSERT(woq_nibble_slot_bytes % 8 == 0);

    mxfp4_woq_repack_src_table src_table{};
    for (int i = 0; i < n_slots; ++i) {
        src_table.ptr[i] = srcs[i];
    }

    const int64_t n_tile_groups_n = N / tile_n_total;
    const int64_t group_bytes     = static_cast<int64_t>(tile_n_total) * (1 + QK_MXFP4 / 2);
    // wg_nibbles dropped to 32 in iteration 3 -- see the per-slot form above.
    constexpr int wg_nibbles      = 32;
    const int     row             = tile_n_total + 1;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> local_acc(sycl::range<1>(32 * row), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, n_slots, blocks_per_row * n_tile_groups_n * wg_nibbles),
                                           sycl::range<3>(1, 1, wg_nibbles)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_xmx_tiled_to_woq_coalesced_nibbles_batched_kernel(
                                 src_table, dst_weight_base, weight_slot_bytes, N, tile_n_total, row, n_tile_groups_n,
                                 group_bytes, get_pointer(local_acc), item);
                         });
    });

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, n_slots, blocks_per_row * n_tile_groups_n * tile_n_total),
                                           sycl::range<3>(1, 1, tile_n_total)),
                         [=](sycl::nd_item<3> item) {
                             repack_mxfp4_xmx_tiled_to_woq_coalesced_scales_batched_kernel(
                                 src_table, dst_weight_base, weight_slot_bytes, woq_nibble_slot_bytes, N, tile_n_total,
                                 n_tile_groups_n, group_bytes, item);
                         });
    });
}

// Host function to launch Q4_0 Coalesced to SoA conversion
void reorder_q4_0_coalesced_to_soa_sycl(const void *    src,
                                        void *          dst,  // SoA format: [all qs bytes][all d values]
                                        int64_t         ne00,
                                        int64_t         ne01,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ne00 % QK4_0 == 0);

    const int blocks_per_row = ne00 / QK4_0;
    const int nrows          = ne01;
    const int tiles_per_row  = (blocks_per_row + MMVQ_COALESCED_TILE_BLOCKS - 1) / MMVQ_COALESCED_TILE_BLOCKS;

    // SoA layout: qs bytes first, then d values
    uint8_t *    dst_qs = (uint8_t *) dst;
    sycl::half * dst_d  = (sycl::half *) ((uint8_t *) dst + nrows * blocks_per_row * (QK4_0 / 2));

    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>(nrows, tiles_per_row * WARP_SIZE), sycl::range<2>(1, WARP_SIZE)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            reorder_q4_0_coalesced_to_soa_kernel((const uint8_t *) src, dst_qs, dst_d, blocks_per_row, nrows, item);
        });
}

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor) {
    // SoA/Coalesced-aware reorder kernels compute d_offset from k parameter.
    // This only works when k == full tensor size. For row slices, use standard kernels.
    const ggml_tensor_extra_gpu * extra =
        dst->src[0]->extra ? static_cast<const ggml_tensor_extra_gpu *>(dst->src[0]->extra) : nullptr;
    const bool use_reorder   = full_tensor && ggml_sycl_layout_is_soa(extra);
    const bool use_coalesced = full_tensor && ggml_sycl_layout_is_coalesced(extra);

    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            if (use_coalesced) {
                return dequantize_row_q4_0_sycl_coalesced;
            } else if (use_reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            if (use_reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            if (use_reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_fp16_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst, bool full_tensor) {
    // SoA/Coalesced-aware reorder kernels compute d_offset from k parameter.
    // This only works when k == full tensor size. For row slices, use standard kernels.
    const ggml_tensor_extra_gpu * extra =
        dst->src[0]->extra ? static_cast<const ggml_tensor_extra_gpu *>(dst->src[0]->extra) : nullptr;
    const bool use_reorder   = full_tensor && ggml_sycl_layout_is_soa(extra);
    const bool use_coalesced = full_tensor && ggml_sycl_layout_is_coalesced(extra);

    switch (type) {
        case GGML_TYPE_Q4_0:
            if (use_coalesced) {
                return dequantize_row_q4_0_sycl_coalesced;
            } else if (use_reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_row_q4_0_sycl;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_sycl;
        case GGML_TYPE_Q4_K:
            if (use_reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_sycl;
        case GGML_TYPE_Q6_K:
            if (use_reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_nc_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            return nullptr;
    }
}
