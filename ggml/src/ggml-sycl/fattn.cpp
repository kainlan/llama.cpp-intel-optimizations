//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn.hpp"

#include "common.hpp"
#include "fattn-debug.hpp"
#include "fattn-esimd-f16.hpp"
#include "fattn-mma.hpp"
#include "fattn-onednn.hpp"
#include "fattn-tile-f16.hpp"
#include "fattn-v2-esimd.hpp"
#include "fattn-v2-partition.hpp"
#include "fattn-vec-f16.hpp"
#include "fattn-vec.hpp"
#include "fattn-xmx-f16-v2.hpp"
#include "fattn-xmx-f16.hpp"
#include "kv-cache-quant.hpp"
#include "l144i-probe.hpp"
#include "sycl-profiling.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Non-inline forwarder — ggml-sycl.cpp's ~ggml_backend_sycl_context() calls this
// without including fattn-xmx-f16-v2.hpp (which drags the SYCL matrix extension).
void ggml_sycl_fattn_xmx_v2_cache_destroy(void * ptr) {
    fattn_xmx_v2_cache_destroy_inline(ptr);
}

static_assert(GGML_SYCL_FATTN_XMX_PACKED_K_D == 64, "Packed-K materializer currently supports D=64 decode K");
static_assert(GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS == XMX_V2_DECODE_BATCH_KV,
              "Packed-K materializer block size must match XMX-v2 decode");
static_assert(GGML_SYCL_FATTN_XMX_PACKED_K_TN == XMX_V2_DECODE_TN,
              "Packed-K materializer N stride must match XMX-v2 decode");
static_assert(GGML_SYCL_FATTN_XMX_PACKED_K_ACTIVE_LANES == XMX_V2_DECODE_ACTIVE_LANES,
              "Packed-K materializer lane compaction must match XMX-v2 decode");
static_assert(GGML_SYCL_FATTN_XMX_PACKED_K_HALFS_PER_BLOCK == fattn_v2_decode_gqa_slm<64>::K_PACKED_ELEMS,
              "Packed-K materializer byte count must match the proven ext_intel_packed ABI");

ggml_sycl_fattn_xmx_packed_k::~ggml_sycl_fattn_xmx_packed_k() {
    reset();
}

ggml_sycl_fattn_xmx_packed_k::ggml_sycl_fattn_xmx_packed_k(ggml_sycl_fattn_xmx_packed_k && other) noexcept {
    *this = std::move(other);
}

ggml_sycl_fattn_xmx_packed_k & ggml_sycl_fattn_xmx_packed_k::operator=(ggml_sycl_fattn_xmx_packed_k && other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();

    alloc       = std::move(other.alloc);
    handle      = std::move(other.handle);
    ready_event = std::move(other.ready_event);
    device      = other.device;
    D           = other.D;
    n_kv        = other.n_kv;
    H_kv        = other.H_kv;
    batch       = other.batch;
    n_blocks    = other.n_blocks;
    total_bytes = other.total_bytes;

    other.alloc       = {};
    other.handle      = {};
    other.ready_event = {};
    other.device      = -1;
    other.D           = 0;
    other.n_kv        = 0;
    other.H_kv        = 0;
    other.batch       = 0;
    other.n_blocks    = 0;
    other.total_bytes = 0;
    return *this;
}

void ggml_sycl_fattn_xmx_packed_k::reset() {
    if (alloc.ptr != nullptr) {
        try {
            ready_event.wait();
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("[SYCL] packed-K materializer event wait failed before free: %s\n", e.what());
        }
        (void) ggml_sycl::unified_free(alloc);
    }

    alloc       = {};
    handle      = {};
    ready_event = {};
    device      = -1;
    D           = 0;
    n_kv        = 0;
    H_kv        = 0;
    batch       = 0;
    n_blocks    = 0;
    total_bytes = 0;
}

namespace {

struct ggml_sycl_fattn_xmx_packed_k_sidecar_entry {
    const void *                    k_base = nullptr;
    size_t                          k_bytes = 0;
    ggml_sycl::mem_handle           k_handle{};
    size_t                          k_handle_hash = 0;
    ggml_sycl_fattn_xmx_packed_k    packed;
};

std::mutex                                                                    g_packed_k_sidecar_mutex;
std::vector<std::unique_ptr<ggml_sycl_fattn_xmx_packed_k_sidecar_entry>>      g_packed_k_sidecars;

static bool ggml_sycl_fattn_xmx_sidecar_enabled() {
    if (std::getenv("GGML_SYCL_PACKED_K_SIDECAR") != nullptr) {
        return true;
    }
    const char * force = std::getenv("GGML_SYCL_FA_FORCE_PATH");
    return force && std::strstr(force, "split-packed") != nullptr;
}

static bool ggml_sycl_fattn_xmx_is_cache_k_tensor(const ggml_tensor * tensor) {
    return tensor && strncmp(tensor->name, "cache_k_l", 9) == 0;
}

static const ggml_tensor * ggml_sycl_fattn_xmx_root_tensor(const ggml_tensor * tensor, size_t * view_offs) {
    if (view_offs) {
        *view_offs = tensor && tensor->view_src ? tensor->view_offs : 0;
    }
    if (!tensor) {
        return nullptr;
    }
    const ggml_tensor * root = tensor;
    while (root->view_src) {
        root = root->view_src;
    }
    return root;
}

static ggml_sycl::mem_handle ggml_sycl_fattn_xmx_root_data_handle(const ggml_tensor * root, int target_device) {
    if (!root || target_device < 0) {
        return {};
    }
    if (root->extra != nullptr && target_device < GGML_SYCL_MAX_DEVICES) {
        auto * extra = static_cast<ggml_tensor_extra_gpu *>(root->extra);
        auto   h     = extra->data_handle[target_device];
        auto   r     = h.resolve(target_device);
        if (r) {
            return h;
        }
    }
    if (root->data == nullptr) {
        return {};
    }
    const auto * info = ggml_sycl::alloc_registry::instance().lookup(root->data);
    if (info && info->type == ggml_sycl::alloc_type::DEVICE) {
        if (info->device_id != target_device) {
            return {};
        }
        return ggml_sycl::mem_handle::from_direct(root->data, GGML_LAYOUT_AOS, true, info->device_id);
    }
    if (info && (info->type == ggml_sycl::alloc_type::HOST_PINNED || info->type == ggml_sycl::alloc_type::SHARED)) {
        return ggml_sycl::mem_handle::from_direct(root->data, GGML_LAYOUT_AOS, false,
                                                  ggml_sycl::mem_handle::HOST_DEVICE);
    }
    return {};
}

static bool ggml_sycl_fattn_xmx_compute_packed_k_bytes(int n_kv,
                                                       int H_kv,
                                                       int batch,
                                                       int * n_blocks,
                                                       size_t * total_bytes) {
    if (n_blocks) {
        *n_blocks = 0;
    }
    if (total_bytes) {
        *total_bytes = 0;
    }
    if (n_kv <= 0 || H_kv <= 0 || batch <= 0) {
        return false;
    }
    const int blocks = (n_kv + GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;
    const size_t sbatch = static_cast<size_t>(batch);
    const size_t sheads = static_cast<size_t>(H_kv);
    const size_t sblocks = static_cast<size_t>(blocks);
    if (sbatch > std::numeric_limits<size_t>::max() / sheads) {
        return false;
    }
    const size_t batch_heads = sbatch * sheads;
    if (sblocks != 0 && batch_heads > std::numeric_limits<size_t>::max() / sblocks) {
        return false;
    }
    const size_t total_blocks = batch_heads * sblocks;
    if (total_blocks > std::numeric_limits<size_t>::max() / GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK) {
        return false;
    }
    if (n_blocks) {
        *n_blocks = blocks;
    }
    if (total_bytes) {
        *total_bytes = total_blocks * GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK;
    }
    return true;
}

template <typename TIdx>
class fattn_xmx_pack_k_set_rows_kernel;

template <typename TIdx>
static sycl::event ggml_sycl_fattn_xmx_submit_set_rows_update(
    const ggml_tensor *                  src0,
    const ggml_tensor *                  src1,
    const void *                         src0_ptr,
    const void *                         index_ptr,
    int                                  row_offset,
    int                                  n_kv,
    int                                  H_kv,
    int                                  batch,
    int                                  n_blocks,
    int                                  heads_per_i02,
    sycl::half *                         packed_ptr,
    dpct::queue_ptr                      stream,
    const sycl::event &                  set_rows_event,
    const sycl::event &                  zero_event,
    const sycl::event &                  previous_sidecar_event,
    bool                                 add_zero_dep,
    bool                                 add_prev_dep) {
    const char * src0_base  = static_cast<const char *>(src0_ptr);
    const char * index_base = static_cast<const char *>(index_ptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t idx_ne11 = src1->ne[1] > 0 ? src1->ne[1] : 1;
    const int64_t idx_ne12 = src1->ne[2] > 0 ? src1->ne[2] : 1;

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];
    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;
    constexpr int block_size = 64;
    const int64_t grid_size = (total_elements + block_size - 1) / block_size;

    const bool add_set_rows_dep = ggml_sycl_should_add_dependency(set_rows_event);

    return stream->submit([&](sycl::handler & cgh) {
        if (add_set_rows_dep) {
            cgh.depends_on(set_rows_event);
        }
        if (add_zero_dep) {
            cgh.depends_on(zero_event);
        }
        if (add_prev_dep) {
            cgh.depends_on(previous_sidecar_event);
        }
        cgh.parallel_for<fattn_xmx_pack_k_set_rows_kernel<TIdx>>(
            sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item) {
                const int64_t i = item.get_global_linear_id();
                if (i >= total_elements) {
                    return;
                }

                const int64_t i03 = i / (ne00 * ne01 * ne02);
                const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
                const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
                const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

                const int head = static_cast<int>(i02) * heads_per_i02 +
                                 static_cast<int>(i00 / GGML_SYCL_FATTN_XMX_PACKED_K_D);
                const int d = static_cast<int>(i00 % GGML_SYCL_FATTN_XMX_PACKED_K_D);

                if (head >= H_kv || i03 >= batch) {
                    return;
                }

                const int64_t i12 = i03 % idx_ne12;
                const int64_t i11 = i02 % idx_ne11;
                const int64_t i10 = i01;
                const int64_t idx_off = i10 * nb10 + i11 * nb11 + i12 * nb12;
                const int64_t dst_row = static_cast<int64_t>(*reinterpret_cast<const TIdx *>(index_base + idx_off));
                const int64_t kv_abs64 = static_cast<int64_t>(row_offset) + dst_row;
                if (kv_abs64 < 0 || kv_abs64 >= n_kv) {
                    return;
                }

                const int kv_abs = static_cast<int>(kv_abs64);
                const int block = kv_abs / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;
                const int kv_local = kv_abs - block * GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;
                if (block < 0 || block >= n_blocks) {
                    return;
                }

                const float src_val =
                    *reinterpret_cast<const float *>(src0_base + i01 * nb01 + i02 * nb02 + i03 * nb03 +
                                                     i00 * static_cast<int64_t>(sizeof(float)));
                const size_t block_base_half =
                    (((static_cast<size_t>(i03) * static_cast<size_t>(H_kv) + static_cast<size_t>(head)) *
                          static_cast<size_t>(n_blocks) +
                      static_cast<size_t>(block)) *
                     GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK) /
                    sizeof(sycl::half);
                const size_t elem_off_half =
                    ggml_sycl_fattn_xmx_packed_k_element_offset_half(kv_local, d);
                if (elem_off_half != SIZE_MAX) {
                    packed_ptr[block_base_half + elem_off_half] = sycl::half(src_val);
                }
            });
    });
}

}  // namespace

ggml_sycl_fattn_xmx_packed_k * ggml_sycl_fattn_xmx_find_packed_k_sidecar(const fattn_params & params,
                                                                         int                  target_device) {
    if (params.K == nullptr || target_device < 0 || params.ne10 != GGML_SYCL_FATTN_XMX_PACKED_K_D ||
        params.ne11 <= 0 || params.ne12 <= 0 || params.ne13 <= 0 || !params.K_handle_valid ||
        params.K_view_offs != 0) {
        return nullptr;
    }
    const int n_partitions = (params.ne11 + GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS - 1) /
                             GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;

    std::lock_guard<std::mutex> lock(g_packed_k_sidecar_mutex);
    for (const auto & entry : g_packed_k_sidecars) {
        const auto & packed = entry->packed;
        if (entry->k_handle.valid() && entry->k_handle_hash == params.K_handle_hash && packed.device == target_device &&
            packed.D == params.ne10 &&
            packed.n_kv >= params.ne11 && packed.H_kv == params.ne12 && packed.batch == params.ne13 &&
            packed.n_blocks >= n_partitions && packed.handle.valid()) {
            return &entry->packed;
        }
    }
    return nullptr;
}

bool ggml_sycl_fattn_xmx_update_packed_k_from_set_rows(const ggml_tensor * dst,
                                                       const ggml_tensor * src0,
                                                       const ggml_tensor * src1,
                                                       int                 target_device,
                                                       const void *        src0_ptr,
                                                       const void *        index_ptr,
                                                       dpct::queue_ptr     stream,
                                                       const sycl::event & set_rows_event,
                                                       sycl::event *       out_event) {
    if (out_event) {
        *out_event = {};
    }
    if (!ggml_sycl_fattn_xmx_sidecar_enabled()) {
        return false;
    }
    auto debug_reject = [&](const char * reason, const ggml_tensor * root = nullptr) {
        if (std::getenv("GGML_SYCL_PACKED_K_DEBUG") || std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG")) {
            static std::atomic<int> count{ 0 };
            const int limit = std::getenv("GGML_SYCL_PACKED_K_DEBUG_LIMIT") ?
                                  std::atoi(std::getenv("GGML_SYCL_PACKED_K_DEBUG_LIMIT")) :
                                  32;
            const int idx = count.fetch_add(1, std::memory_order_relaxed);
            if (idx < limit) {
                fprintf(stderr,
                        "[SYCL] packed-K SET_ROWS sidecar reject[%d/%d]: %s dst=%s root=%s "
                        "dst_type=%d src0_type=%d src1_type=%d target=%d\n",
                        idx + 1, limit, reason, dst ? dst->name : "(null)", root ? root->name : "(null)",
                        dst ? (int) dst->type : -1, src0 ? (int) src0->type : -1, src1 ? (int) src1->type : -1,
                        target_device);
            }
        }
        return false;
    };
    if (!dst || !src0 || !src1 || !src0_ptr || !index_ptr || !stream || target_device < 0) {
        return debug_reject("missing-input");
    }
    if (dst->type != GGML_TYPE_F16 || src0->type != GGML_TYPE_F32 ||
        (src1->type != GGML_TYPE_I64 && src1->type != GGML_TYPE_I32)) {
        return debug_reject("unsupported-types");
    }

    size_t view_offs = 0;
    const ggml_tensor * root = ggml_sycl_fattn_xmx_root_tensor(dst, &view_offs);
    if (!ggml_sycl_fattn_xmx_is_cache_k_tensor(root) || root->type != GGML_TYPE_F16 || src0->ne[0] <= 0 ||
        src0->ne[0] % GGML_SYCL_FATTN_XMX_PACKED_K_D != 0 || dst->ne[0] != src0->ne[0] || dst->ne[1] <= 0 ||
        src0->ne[2] <= 0 || dst->ne[3] <= 0) {
        return debug_reject("not-cache-k-d64", root);
    }

    const int heads_per_i02 = static_cast<int>(src0->ne[0] / GGML_SYCL_FATTN_XMX_PACKED_K_D);
    const int packed_H_kv = static_cast<int>(src0->ne[2]) * heads_per_i02;
    if (packed_H_kv <= 0 || src0->ne[3] > dst->ne[3]) {
        return debug_reject("src-shape-exceeds-k-cache", root);
    }

    int row_offset = 0;
    if (root != dst) {
        if (root->nb[1] <= 0 || view_offs % static_cast<size_t>(root->nb[1]) != 0) {
            return debug_reject("bad-view-row-alignment", root);
        }
        row_offset = static_cast<int>(view_offs / static_cast<size_t>(root->nb[1]));
        if (row_offset < 0 || row_offset >= dst->ne[1]) {
            return debug_reject("bad-view-row-offset", root);
        }
    }

    const auto * info = ggml_sycl::alloc_registry::instance().lookup(root->data);
    if (!info || info->type != ggml_sycl::alloc_type::DEVICE || info->device_id != target_device) {
        return debug_reject("k-cache-not-device-resident", root);
    }
    ggml_sycl::mem_handle root_handle = ggml_sycl_fattn_xmx_root_data_handle(root, target_device);
    if (!root_handle.valid()) {
        return debug_reject("missing-k-cache-handle", root);
    }
    const size_t root_handle_hash = root_handle.hash();

    const int n_kv = static_cast<int>(dst->ne[1]);
    const int H_kv = packed_H_kv;
    const int batch = static_cast<int>(dst->ne[3]);
    int n_blocks = 0;
    size_t total_bytes = 0;
    if (!ggml_sycl_fattn_xmx_compute_packed_k_bytes(n_kv, H_kv, batch, &n_blocks, &total_bytes)) {
        return debug_reject("packed-size-overflow", root);
    }

    ggml_sycl_fattn_xmx_packed_k_sidecar_entry * entry = nullptr;
    sycl::event zero_event;
    bool add_zero_dep = false;
    bool add_prev_dep = false;
    {
        std::lock_guard<std::mutex> lock(g_packed_k_sidecar_mutex);
        for (const auto & candidate : g_packed_k_sidecars) {
            if (candidate->k_handle.valid() && candidate->k_handle_hash == root_handle_hash &&
                candidate->packed.device == target_device) {
                entry = candidate.get();
                break;
            }
        }
        if (entry == nullptr) {
            auto owned = std::make_unique<ggml_sycl_fattn_xmx_packed_k_sidecar_entry>();
            owned->k_base = root->data;
            owned->k_bytes = ggml_nbytes(root);
            owned->k_handle = root_handle;
            owned->k_handle_hash = root_handle_hash;
            entry = owned.get();
            g_packed_k_sidecars.push_back(std::move(owned));
        }
        entry->k_base = root->data;
        entry->k_bytes = ggml_nbytes(root);
        entry->k_handle = root_handle;
        entry->k_handle_hash = root_handle_hash;

        ggml_sycl_fattn_xmx_packed_k & packed = entry->packed;
        const bool reuse_alloc = packed.alloc.ptr != nullptr && packed.device == target_device &&
                                 packed.D == GGML_SYCL_FATTN_XMX_PACKED_K_D && packed.H_kv == H_kv &&
                                 packed.batch == batch && packed.n_kv >= n_kv && packed.n_blocks >= n_blocks &&
                                 packed.alloc.tier == ggml_sycl::alloc_tier::DEVICE_VRAM &&
                                 packed.alloc.size >= total_bytes;
        if (!reuse_alloc) {
            packed.reset();

            ggml_sycl::alloc_request req{};
            req.queue                               = stream;
            req.device                              = target_device;
            req.size                                = total_bytes;
            req.intent.role                         = ggml_sycl::alloc_role::KV;
            req.intent.category                     = ggml_sycl::runtime_category::KV_CACHE;
            req.intent.cohort_id                    = "fattn_xmx_packed_k_sidecar";
            req.intent.constraints.must_device      = true;
            req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::KV;
            if (!ggml_sycl::unified_alloc(req, &packed.alloc) || packed.alloc.ptr == nullptr ||
                packed.alloc.tier != ggml_sycl::alloc_tier::DEVICE_VRAM) {
                packed.reset();
                return debug_reject("sidecar-alloc-failed", root);
            }
            packed.handle = packed.alloc.as_mem_handle();
            zero_event = stream->memset(packed.alloc.ptr, 0, total_bytes);
            add_zero_dep = ggml_sycl_should_add_dependency(zero_event);
        } else {
            try {
                add_prev_dep = ggml_sycl_should_add_dependency(packed.ready_event);
            } catch (...) {
                add_prev_dep = true;
            }
        }

        packed.device      = target_device;
        packed.D           = GGML_SYCL_FATTN_XMX_PACKED_K_D;
        packed.n_kv        = n_kv;
        packed.H_kv        = H_kv;
        packed.batch       = batch;
        packed.n_blocks    = n_blocks;
        packed.total_bytes = packed.alloc.size;

        try {
            sycl::event update_event =
                src1->type == GGML_TYPE_I64 ?
                    ggml_sycl_fattn_xmx_submit_set_rows_update<int64_t>(
                        src0, src1, src0_ptr, index_ptr, row_offset, n_kv, H_kv, batch, n_blocks,
                        heads_per_i02, static_cast<sycl::half *>(packed.alloc.ptr), stream, set_rows_event, zero_event,
                        packed.ready_event, add_zero_dep, add_prev_dep) :
                    ggml_sycl_fattn_xmx_submit_set_rows_update<int32_t>(
                        src0, src1, src0_ptr, index_ptr, row_offset, n_kv, H_kv, batch, n_blocks,
                        heads_per_i02, static_cast<sycl::half *>(packed.alloc.ptr), stream, set_rows_event, zero_event,
                        packed.ready_event, add_zero_dep, add_prev_dep);
            packed.ready_event = update_event;
            if (out_event) {
                *out_event = update_event;
            }
        } catch (const sycl::exception & e) {
            GGML_LOG_WARN("[SYCL] packed-K SET_ROWS sidecar update submit failed: %s\n", e.what());
            if (!reuse_alloc) {
                packed.reset();
            }
            return false;
        }
    }

    return true;
}

void ggml_sycl_fattn_xmx_unregister_packed_k_range(const void * ptr, size_t size) {
    if (!ptr || size == 0) {
        return;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t end = begin + size;
    std::lock_guard<std::mutex> lock(g_packed_k_sidecar_mutex);
    auto it = g_packed_k_sidecars.begin();
    while (it != g_packed_k_sidecars.end()) {
        const uintptr_t k = reinterpret_cast<uintptr_t>((*it)->k_base);
        if (k >= begin && k < end) {
            it = g_packed_k_sidecars.erase(it);
        } else {
            ++it;
        }
    }
}

// Kernel names for VTune profiling
class fattn_v2_fill_block_table_kernel;
class fattn_v2_fill_block_table_runtime_kernel;
class fattn_v2_fill_seq_lens_kernel;

// =============================================================================
// V2 Partitioned Attention Compile-Time Switch
// =============================================================================
// When enabled (1), V2 multi-partition algorithm is available for sequences > 512
// When disabled (0), always uses standard XMX flash attention
// Can also be set via cmake: -DGGML_SYCL_FA_V2_ENABLED=OFF
//
// NOTE: V2 kernel now uses stride-based addressing (nb11, nb12, nb21, nb22)
// matching the XMX kernel pattern for compatibility with llama.cpp's KV cache layout.
#ifndef GGML_SYCL_FA_V2_ENABLED
#    define GGML_SYCL_FA_V2_ENABLED 1  // Enabled: stride-based addressing implemented
#endif

// =============================================================================
// Paged Attention V2 Configuration
// =============================================================================
// V2 uses multi-partition algorithm for long sequences (>512 tokens)
// This enables O(n) memory complexity for O(n²) attention
//
// Enable with environment variable: GGML_SYCL_PAGED_V2=1
//
// V2 modes:
// 1. With paged KV layout: Uses actual block tables for logical->physical mapping
// 2. With contiguous KV layout (auto-V2): Generates identity block table with block_size=1
//
// Auto-V2 allows V2 partitioning benefits without requiring paged KV cache changes

#if GGML_SYCL_FA_V2_ENABLED

static bool g_sycl_paged_v2_enabled     = false;
static bool g_sycl_paged_v2_initialized = false;

static void init_paged_v2_config() {
    if (g_sycl_paged_v2_initialized) {
        return;
    }
    g_sycl_paged_v2_initialized = true;

    const char * env = std::getenv("GGML_SYCL_PAGED_V2");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_paged_v2_enabled = true;
        fprintf(stderr, "[SYCL] Paged Attention V2 enabled for long sequences (>512 tokens)\n");
        fprintf(stderr, "[SYCL]   Auto-V2 mode: identity block table with block_size=1\n");
    }
}

// =============================================================================
// Auto-V2: Thread-local buffers for identity block table (contiguous KV mode)
// =============================================================================
// When V2 is enabled but paged KV is not active, we generate an identity block
// table where block_table[i] = i and block_size = 1. This allows V2 partitioning
// to work with the existing contiguous 3D KV cache layout.

struct v2_auto_buffers {
    int32_t * block_table          = nullptr;  // Identity block table [num_seqs, max_blocks]
    int32_t * seq_lens             = nullptr;  // Sequence lengths [num_seqs]
    size_t    block_table_capacity = 0;        // Current capacity in elements
    size_t    seq_lens_capacity    = 0;        // Current capacity in elements

    // Persistent temp buffers for V2 kernel (avoids malloc/free during graph recording)
    // Layout: [exp_sums | max_logits | tmp_out]
    void * temp_buf          = nullptr;
    size_t temp_buf_capacity = 0;  // Current capacity in bytes

    sycl::queue * alloc_queue = nullptr;

    ~v2_auto_buffers() {
        if (block_table && alloc_queue) {
            sycl::free(block_table, *alloc_queue);
        }
        if (seq_lens && alloc_queue) {
            sycl::free(seq_lens, *alloc_queue);
        }
        if (temp_buf && alloc_queue) {
            sycl::free(temp_buf, *alloc_queue);
        }
    }
};

static thread_local v2_auto_buffers g_v2_auto;

// Thread-local buffers for multi-sequence flash attention
// Freed on thread exit to avoid leaking device memory.
static std::atomic<bool> g_fattn_shutting_down{ false };
static std::atomic<bool> g_fattn_atexit_registered{ false };
static std::atomic<int>  g_seq_id_buffer_instances{ 0 };
static std::atomic<int>  g_seq_id_buffer_allocs{ 0 };

// atexit handler to prevent SYCL cleanup during static destruction
static void fattn_atexit_handler() {
    g_fattn_shutting_down.store(true);
}

static void register_fattn_atexit() {
    bool expected = false;
    if (g_fattn_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(fattn_atexit_handler);
    }
}

struct tl_seq_id_buffers {
    int32_t *     q_seq_ids_dev        = nullptr;
    int32_t *     kv_seq_ids_dev       = nullptr;
    size_t        q_seq_ids_size       = 0;
    size_t        kv_seq_ids_size      = 0;
    int32_t *     seq_q_offsets_dev    = nullptr;
    int32_t *     seq_kv_offsets_dev   = nullptr;
    size_t        seq_offsets_capacity = 0;
    sycl::queue * alloc_queue          = nullptr;

    tl_seq_id_buffers() {
        register_fattn_atexit();
        g_seq_id_buffer_instances.fetch_add(1);
    }

    ~tl_seq_id_buffers() {
        free_all();
        g_seq_id_buffer_instances.fetch_sub(1);
    }

    void free_ptr(int32_t *& ptr, size_t bytes) {
        if (ptr && alloc_queue) {
            sycl::free(ptr, *alloc_queue);
            ptr = nullptr;
            g_seq_id_buffer_allocs.fetch_sub(1);
        }
    }

    int32_t * alloc_ptr(size_t count, sycl::queue * stream) {
        int32_t * ptr = ggml_sycl_malloc_device_t<int32_t>(count, *stream, "fattn_block_table");
        if (ptr) {
            g_seq_id_buffer_allocs.fetch_add(1);
        }
        return ptr;
    }

    void free_all() {
        if (g_fattn_shutting_down.load(std::memory_order_acquire)) {
            return;
        }
        free_ptr(q_seq_ids_dev, q_seq_ids_size);
        free_ptr(kv_seq_ids_dev, kv_seq_ids_size);
        free_ptr(seq_q_offsets_dev, seq_offsets_capacity);
        free_ptr(seq_kv_offsets_dev, seq_offsets_capacity);
        q_seq_ids_size       = 0;
        kv_seq_ids_size      = 0;
        seq_offsets_capacity = 0;
    }
};

static thread_local tl_seq_id_buffers g_tl_seq_buffers;

// =============================================================================
// llama.cpp-15li2 CRIT-1: per-op staging for FA weight sources (attn_sinks).
// =============================================================================
// GPT-OSS adds a per-layer `attn_sinks` weight tensor
// (src/llama-model.cpp:8082) that flows into FLASH_ATTN_EXT as src[4].  When
// the placement planner puts a GPT-OSS layer's dense weights on host (e.g.
// GGML_SYCL_VRAM_BUDGET_PCT=30), `attn_sinks` is host-resident.  The FA
// dispatch (below) resolves src[4] via ggml_sycl_resolve_tensor_ptr and
// passes the pointer straight to the XMX kernel as `params.sinks`.
//
// Host-pinned USM (sycl::malloc_host, which is what the host arena uses for
// weights) is GPU-addressable via PCIe zero-copy, so the invariant usually
// holds.  But a future change that puts any FA weight source in non-USM
// memory (e.g. a plain malloc fallback) would crash the kernel with a GPU
// page fault, and the layer-plan-based dispatch cannot see it.
//
// Belt-and-braces fix: at FA kernel submission time, if any weight-rooted
// src pointer resolves to something other than `sycl::usm::alloc::device`,
// stage a temporary device copy.  Host-pinned USM (alloc::host) could be
// used directly via PCIe, but explicitly staging to device gives the XMX
// kernel uniform fast-path memory access and eliminates the hazard
// entirely.
//
// Implementation: thread-local reusable device staging buffer (pattern
// mirrors g_tl_seq_buffers).  Memcpy happens on the inference stream, so
// it is correctly ordered before the FA kernel that reads it (in-order
// queue) without an explicit barrier.
struct tl_fattn_weight_stage {
    void *        sinks_dev      = nullptr;
    size_t        sinks_capacity = 0;
    void *        mask_dev       = nullptr;
    size_t        mask_capacity  = 0;
    sycl::queue * alloc_queue    = nullptr;

    tl_fattn_weight_stage() { register_fattn_atexit(); }

    ~tl_fattn_weight_stage() { free_all(); }

    void free_all() {
        if (g_fattn_shutting_down.load(std::memory_order_acquire)) {
            return;
        }
        if (sinks_dev && alloc_queue) {
            sycl::free(sinks_dev, *alloc_queue);
            sinks_dev      = nullptr;
            sinks_capacity = 0;
        }
        if (mask_dev && alloc_queue) {
            sycl::free(mask_dev, *alloc_queue);
            mask_dev      = nullptr;
            mask_capacity = 0;
        }
    }
};

static thread_local tl_fattn_weight_stage g_tl_fattn_weight_stage;

// Returns a device-addressable pointer for `src` on `stream`.  If the
// resolved pointer is already sycl::usm::alloc::device, returns it as-is.
// Otherwise copies the tensor's bytes into the thread-local device staging
// buffer `slot` (growing it if needed) and returns the staged pointer.
//
// Returns nullptr on allocation failure; caller may fall back to the
// zero-copy host-pinned pointer (which is GPU-accessible but slower).
//
// `slot_ptr` / `slot_capacity` reference the staging-buffer slot to reuse
// across calls (one slot per tensor role).
static const char * ggml_sycl_fattn_stage_weight_src(const ggml_tensor * src,
                                                     int                 device,
                                                     sycl::queue *       stream,
                                                     const char *        role,
                                                     void *&             slot_ptr,
                                                     size_t &            slot_capacity) {
    if (!src || !stream) {
        return nullptr;
    }
    void * resolved = ggml_sycl_resolve_tensor_ptr(const_cast<ggml_tensor *>(src), device);
    if (!resolved) {
        return nullptr;
    }
    // Fast path: already on device — pass through.
    sycl::usm::alloc alloc_kind = sycl::usm::alloc::unknown;
    try {
        alloc_kind = ggml_sycl_get_alloc_type(resolved);
    } catch (...) {
        // Query failure: treat as unknown; fall through to staging.
    }
    if (alloc_kind == sycl::usm::alloc::device) {
        return static_cast<const char *>(resolved);
    }

    const size_t bytes = static_cast<size_t>(ggml_nbytes(src));
    if (bytes == 0) {
        return static_cast<const char *>(resolved);
    }

    // Stage: grow the slot if needed, then memcpy on the inference stream.
    auto & stage = g_tl_fattn_weight_stage;
    if (stage.alloc_queue && stage.alloc_queue != stream) {
        // Queue changed (unlikely during steady-state inference): free both
        // slots and reallocate on the new queue.
        stage.free_all();
    }
    stage.alloc_queue = stream;

    if (bytes > slot_capacity) {
        if (slot_ptr) {
            sycl::free(slot_ptr, *stream);
            slot_ptr      = nullptr;
            slot_capacity = 0;
        }
        slot_ptr = ggml_sycl_malloc_device(bytes, *stream, "fattn_weight_stage");
        if (!slot_ptr) {
            static bool warned = false;
            if (!warned) {
                GGML_LOG_WARN(
                    "[SYCL] fattn weight staging: failed to allocate %zu bytes for %s; "
                    "falling back to host-pinned zero-copy (may be slower)\n",
                    bytes, role);
                warned = true;
            }
            return static_cast<const char *>(resolved);  // Host-pinned USM, still GPU-readable.
        }
        slot_capacity = bytes;
    }

    // In-order queue ensures this memcpy completes before the FA kernel
    // submission that follows.  Tiny copy (attn_sinks is n_head * f32 =
    // 256 B for GPT-OSS 20B), so cost is dominated by submission overhead
    // not bandwidth.
    stream->memcpy(slot_ptr, resolved, bytes);

    static std::atomic<int> staged_log_left{ 8 };
    if (staged_log_left.fetch_sub(1) > 0) {
        GGML_LOG_INFO("[SYCL] fattn staged weight-source '%s' (%s, %zu B) from alloc_kind=%d to device\n",
                      src->name[0] ? src->name : "?", role, bytes, static_cast<int>(alloc_kind));
    }
    return static_cast<const char *>(slot_ptr);
}

// Test hooks for verifying thread-local cleanup.
extern "C" int ggml_sycl_test_seq_id_buffer_instances() {
    return g_seq_id_buffer_instances.load();
}

extern "C" int ggml_sycl_test_seq_id_buffer_allocs() {
    return g_seq_id_buffer_allocs.load();
}

extern "C" void ggml_sycl_test_fattn_set_shutdown(int value) {
    g_fattn_shutting_down.store(value != 0, std::memory_order_release);
}

extern "C" void ggml_sycl_test_seq_id_buffers_free_all() {
    g_tl_seq_buffers.free_all();
}

extern "C" bool ggml_sycl_test_seq_id_buffers_touch(sycl::queue * stream) {
    if (!stream) {
        return false;
    }
    if (g_tl_seq_buffers.alloc_queue && g_tl_seq_buffers.alloc_queue != stream) {
        g_tl_seq_buffers.free_all();
    }
    g_tl_seq_buffers.alloc_queue = stream;

    if (!g_tl_seq_buffers.q_seq_ids_dev) {
        g_tl_seq_buffers.q_seq_ids_dev  = g_tl_seq_buffers.alloc_ptr(1, stream);
        g_tl_seq_buffers.q_seq_ids_size = sizeof(int32_t);
    }
    if (!g_tl_seq_buffers.kv_seq_ids_dev) {
        g_tl_seq_buffers.kv_seq_ids_dev  = g_tl_seq_buffers.alloc_ptr(1, stream);
        g_tl_seq_buffers.kv_seq_ids_size = sizeof(int32_t);
    }
    if (!g_tl_seq_buffers.seq_q_offsets_dev) {
        g_tl_seq_buffers.seq_q_offsets_dev = g_tl_seq_buffers.alloc_ptr(1, stream);
    }
    if (!g_tl_seq_buffers.seq_kv_offsets_dev) {
        g_tl_seq_buffers.seq_kv_offsets_dev = g_tl_seq_buffers.alloc_ptr(1, stream);
    }

    return g_tl_seq_buffers.q_seq_ids_dev && g_tl_seq_buffers.kv_seq_ids_dev && g_tl_seq_buffers.seq_q_offsets_dev &&
           g_tl_seq_buffers.seq_kv_offsets_dev;
}

// Pre-allocate V2 buffers before SYCL graph recording starts.
// This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
// Called from ggml-sycl.cpp before graph recording.
//
// Parameters are estimated from cgraph FLASH_ATTN_EXT ops - we allocate for the maximum
// expected context length found in the graph.
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Initialize V2 config if not already done
    init_paged_v2_config();

    // Skip if V2 is not enabled
    if (!g_sycl_paged_v2_enabled) {
        return;
    }

    // Find maximum context length from FLASH_ATTN_EXT ops in graph
    int max_context_len  = 0;
    int max_num_heads    = 0;
    int max_num_kv_heads = 0;
    int max_head_dim     = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        // K tensor has shape [D, n_kv, n_kv_heads] where n_kv is context length
        const ggml_tensor * K = node->src[1];
        if (!K) {
            continue;
        }

        const int ctx_len      = (int) K->ne[1];  // n_kv (current KV cache length)
        const int num_kv_heads = (int) K->ne[2];
        const int head_dim     = (int) K->ne[0];

        // Q tensor has shape [D, n_q, n_heads]
        const ggml_tensor * Q         = node->src[0];
        const int           num_heads = Q ? (int) Q->ne[2] : num_kv_heads;

        if (ctx_len > max_context_len) {
            max_context_len = ctx_len;
        }
        if (num_heads > max_num_heads) {
            max_num_heads = num_heads;
        }
        if (num_kv_heads > max_num_kv_heads) {
            max_num_kv_heads = num_kv_heads;
        }
        if (head_dim > max_head_dim) {
            max_head_dim = head_dim;
        }
    }

    // No FLASH_ATTN_EXT ops found, nothing to pre-allocate
    if (max_context_len == 0) {
        return;
    }

    // Only pre-allocate if context is long enough to use V2
    // V2 threshold is 512 tokens (from should_use_paged_attention_v2)
    if (max_context_len <= 512) {
        return;
    }

    // Calculate buffer sizes for auto-V2 mode
    // For auto-V2: block_size = 1, so max_blocks_per_seq = max_context_len
    const int num_seqs         = 1;  // Decode mode typically has 1 query per sequence
    const int block_table_size = num_seqs * max_context_len;
    const int seq_lens_size    = num_seqs;

    // Calculate temp buffer size
    const size_t temp_size = paged_attention_v2_temp_size(num_seqs, max_num_heads, max_context_len, max_head_dim);

    // Pre-allocate block_table if needed
    if (!g_v2_auto.block_table || g_v2_auto.alloc_queue != ctx.stream() ||
        g_v2_auto.block_table_capacity < (size_t) block_table_size) {
        if (g_v2_auto.block_table && g_v2_auto.alloc_queue) {
            sycl::free(g_v2_auto.block_table, *g_v2_auto.alloc_queue);
        }
        g_v2_auto.block_table =
            ggml_sycl_malloc_device_t<int32_t>(block_table_size, *ctx.stream(), "fattn_block_table");
        g_v2_auto.block_table_capacity = block_table_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] block_table allocated: %d elements\n", block_table_size);
    }

    // Pre-allocate seq_lens if needed
    if (!g_v2_auto.seq_lens || g_v2_auto.alloc_queue != ctx.stream() ||
        g_v2_auto.seq_lens_capacity < (size_t) seq_lens_size) {
        if (g_v2_auto.seq_lens && g_v2_auto.alloc_queue) {
            sycl::free(g_v2_auto.seq_lens, *g_v2_auto.alloc_queue);
        }
        g_v2_auto.seq_lens = ggml_sycl_malloc_device_t<int32_t>(seq_lens_size, *ctx.stream(), "fattn_seq_lens");
        g_v2_auto.seq_lens_capacity = seq_lens_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] seq_lens allocated: %d elements\n", seq_lens_size);
    }

    // Pre-allocate temp buffer if needed
    if (!g_v2_auto.temp_buf || g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.temp_buf_capacity < temp_size) {
        if (g_v2_auto.temp_buf && g_v2_auto.alloc_queue) {
            sycl::free(g_v2_auto.temp_buf, *g_v2_auto.alloc_queue);
        }
        g_v2_auto.temp_buf          = ggml_sycl_malloc_device_t<uint8_t>(temp_size, *ctx.stream(), "fattn_temp");
        g_v2_auto.temp_buf_capacity = temp_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] temp_buf allocated: %zu bytes\n", temp_size);
    }

    // Update alloc_queue
    g_v2_auto.alloc_queue = ctx.stream();

    // Pre-fill identity block table (needed for graph recording)
    // This is a kernel launch, which is fine during graph recording prep
    ctx.stream()->parallel_for<class fattn_v2_fill_block_table_kernel>(
        sycl::range<1>(block_table_size), [block_table_ptr = g_v2_auto.block_table](sycl::id<1> idx) {
            block_table_ptr[idx] = static_cast<int32_t>(idx[0]);
        });

    // Wait for pre-allocation to complete before graph recording starts
    ctx.stream()->wait();

    fprintf(stderr, "[SYCL] V2 buffers pre-allocated for graph recording (ctx=%d, heads=%d, D=%d)\n", max_context_len,
            max_num_heads, max_head_dim);
}

#else   // GGML_SYCL_FA_V2_ENABLED == 0

// Stub when V2 is disabled
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    (void) ctx;
    (void) cgraph;
    // V2 disabled at compile time, nothing to pre-allocate
}

#endif  // GGML_SYCL_FA_V2_ENABLED

// =============================================================================
// FP8 KV Cache Configuration
// =============================================================================
// FP8 quantization provides 2x memory reduction for KV cache
//
// Enable with environment variable: GGML_SYCL_KV_FP8=1
// Requires FP8-compatible KV cache allocation

static bool g_sycl_kv_fp8_enabled     = false;
static bool g_sycl_kv_fp8_initialized = false;

static void init_kv_fp8_config() {
    if (g_sycl_kv_fp8_initialized) {
        return;
    }
    g_sycl_kv_fp8_initialized = true;

    const char * env = std::getenv("GGML_SYCL_KV_FP8");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_kv_fp8_enabled = true;
        fprintf(stderr, "[SYCL] FP8 KV cache quantization enabled (2x memory savings)\n");
    }
}

// =============================================================================
// ESIMD Flash Attention Configuration
// =============================================================================
// ESIMD uses explicit SIMD operations for potentially better decode performance
// compared to XMX joint_matrix operations.
//
// Enabled by default. Disable with: GGML_SYCL_FA_ESIMD=0

static bool g_sycl_fa_esimd_enabled           = false;
static bool g_sycl_fa_esimd_initialized       = false;
static bool g_sycl_fa_safe_decode_initialized = false;
static bool g_sycl_fa_safe_decode_enabled     = true;
static bool g_sycl_fa_safe_decode_forced      = false;

static void init_fa_esimd_config() {
    if (g_sycl_fa_esimd_initialized) {
        return;
    }
    g_sycl_fa_esimd_initialized = true;

    // ESIMD FA is enabled by default when available (7-8% decode speedup)
    // Disable with GGML_SYCL_FA_ESIMD=0
    const char * env          = std::getenv("GGML_SYCL_FA_ESIMD");
    bool         enable_esimd = true;  // Default to enabled
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        enable_esimd = false;
    }

    if (enable_esimd && fattn_esimd_f16_available()) {
        g_sycl_fa_esimd_enabled = true;
        GGML_SYCL_DEBUG("[SYCL] ESIMD Flash Attention enabled for decode\n");
    }
}

// =============================================================================
// oneDNN graph SDPA path configuration (Phase 3)
// =============================================================================
// Routes eligible models (no sinks, no softcap, f16 KV) through oneDNN graph SDPA.
// Enabled by default. Disable with: GGML_SYCL_FA_ONEDNN=0
#if GGML_SYCL_DNNL
static bool g_sycl_fa_onednn_enabled     = true;
static bool g_sycl_fa_onednn_initialized = false;

static void init_fa_onednn_config() {
    if (g_sycl_fa_onednn_initialized) {
        return;
    }
    g_sycl_fa_onednn_initialized = true;
    const char * env             = std::getenv("GGML_SYCL_FA_ONEDNN");
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        g_sycl_fa_onednn_enabled = false;
        fprintf(stderr, "[SYCL] oneDNN SDPA path disabled (GGML_SYCL_FA_ONEDNN=0)\n");
    }
    // FA_NO_XMX=1 is an ergonomic escape hatch users reach for when XMX
    // produces wrong output (e.g. during an XMX-kernel correctness regression).
    // oneDNN's fused SDPA internally dispatches to DPAS / XMX on Xe2, so
    // leaving it enabled defeats the user's intent. Force-disable oneDNN
    // whenever FA_NO_XMX is set, independent of the FA_ONEDNN gate so the
    // user doesn't have to unset two env vars for one behaviour.
    const char * no_xmx_env = std::getenv("GGML_SYCL_FA_NO_XMX");
    if (g_sycl_fa_onednn_enabled && no_xmx_env && std::atoi(no_xmx_env) != 0) {
        g_sycl_fa_onednn_enabled = false;
        fprintf(stderr, "[SYCL] oneDNN SDPA path disabled (GGML_SYCL_FA_NO_XMX=1 implies oneDNN-off)\n");
    }
}
#endif  // GGML_SYCL_DNNL

static void init_fa_safe_decode_config() {
    if (g_sycl_fa_safe_decode_initialized) {
        return;
    }
    g_sycl_fa_safe_decode_initialized = true;
    const char * env                  = std::getenv("GGML_SYCL_FA_SAFE_DECODE");
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        g_sycl_fa_safe_decode_enabled = false;
    } else if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_fa_safe_decode_enabled = true;
        g_sycl_fa_safe_decode_forced  = true;
    }
}

ggml_sycl_fattn_decode_policy ggml_sycl_fattn_fast_decode_policy(const fattn_params & params, int D) {
    if (params.ne01 > 1) {
        return { false, ggml_sycl_fattn_decode_policy_reason::NOT_SINGLE_QUERY };
    }
    if (params.kv_is_fp8) {
        return { false, ggml_sycl_fattn_decode_policy_reason::FP8_KV_UNSUPPORTED };
    }
    // The ESIMD f16 decode kernel implements the stateless attention modifiers
    // directly: attention sinks after the KV loop, optional logit softcap, and
    // ALiBi slope scaling. Keep routing descriptors below conservative because
    // they alter KV addressing or sequence ownership and need separate proof.
    if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
        params.seq_lens != nullptr) {
        return { false, ggml_sycl_fattn_decode_policy_reason::PAGED_UNSUPPORTED };
    }
    if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr) {
        return { false, ggml_sycl_fattn_decode_policy_reason::MULTI_SEQ_UNSUPPORTED };
    }
    if (params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
        return { false, ggml_sycl_fattn_decode_policy_reason::SEQ_IDS_UNSUPPORTED };
    }
    if (params.multi_token_decode || params.q_positions != nullptr) {
        return { false, ggml_sycl_fattn_decode_policy_reason::MULTI_TOKEN_DECODE_UNSUPPORTED };
    }
    if (!fattn_vec_supports_head_dim(D)) {
        return { false, ggml_sycl_fattn_decode_policy_reason::HEAD_DIM_UNSUPPORTED };
    }
    if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0) {
        return { false, ggml_sycl_fattn_decode_policy_reason::HEAD_RATIO_UNSUPPORTED };
    }
    if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
        return { false, ggml_sycl_fattn_decode_policy_reason::Q_TYPE_UNSUPPORTED };
    }
    if (params.K_type != GGML_TYPE_F16) {
        return { false, ggml_sycl_fattn_decode_policy_reason::K_TYPE_UNSUPPORTED };
    }
    if (params.V_type != GGML_TYPE_F16) {
        return { false, ggml_sycl_fattn_decode_policy_reason::V_TYPE_UNSUPPORTED };
    }
    return { true, ggml_sycl_fattn_decode_policy_reason::OK };
}

static ggml_sycl_fattn_xmx_decode_kv_layout_plan ggml_sycl_fattn_xmx_decode_kv_reject(
    const fattn_params &                        params,
    int                                         D,
    ggml_sycl_fattn_xmx_decode_kv_layout_reason reason) {
    return {
        ggml_sycl_fattn_xmx_decode_kv_layout_kind::REJECT, reason, D, params.ne02, params.ne12, 0, 0, 0, 0, 0, 0, 0, 0
    };
}

ggml_sycl_fattn_xmx_decode_kv_layout_plan ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(
    const fattn_params &                       params,
    int                                        D,
    const ggml_sycl_fattn_xmx_decode_kv_caps & caps) {
    if (params.ne01 != 1) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::NOT_SINGLE_QUERY);
    }
    if (params.kv_is_fp8) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::FP8_KV_UNSUPPORTED);
    }
    if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
        params.seq_lens != nullptr) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::PAGED_UNSUPPORTED);
    }
    if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
        params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::MULTI_SEQ_UNSUPPORTED);
    }
    if (params.multi_token_decode || params.q_positions != nullptr) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(
            params, D, ggml_sycl_fattn_xmx_decode_kv_layout_reason::MULTI_TOKEN_DECODE_UNSUPPORTED);
    }
    if (D != 64) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::HEAD_DIM_UNSUPPORTED);
    }
    if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(
            params, D, ggml_sycl_fattn_xmx_decode_kv_layout_reason::HEAD_RATIO_UNSUPPORTED);
    }
    const int n_rep = params.ne02 / params.ne12;
    if (n_rep <= 0 || n_rep > XMX_V2_DECODE_GQA_MAX) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::GQA_RATIO_UNSUPPORTED);
    }
    if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::Q_TYPE_UNSUPPORTED);
    }
    if (params.K_type != GGML_TYPE_F16) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::K_TYPE_UNSUPPORTED);
    }
    if (params.V_type != GGML_TYPE_F16) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(params, D,
                                                    ggml_sycl_fattn_xmx_decode_kv_layout_reason::V_TYPE_UNSUPPORTED);
    }
    if (!caps.k_device_resident || !caps.v_device_resident) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(
            params, D, ggml_sycl_fattn_xmx_decode_kv_layout_reason::KV_NOT_DEVICE_RESIDENT);
    }
    if (!caps.m1n64_k16_supported) {
        return ggml_sycl_fattn_xmx_decode_kv_reject(
            params, D, ggml_sycl_fattn_xmx_decode_kv_layout_reason::DEVICE_XMX_M1N64_UNSUPPORTED);
    }

    const size_t required_slm_bytes       = fattn_v2_decode_gqa_slm<64>::TOTAL * sizeof(sycl::half);
    const size_t source_k_bytes_per_block = XMX_V2_DECODE_BATCH_KV * D * sizeof(sycl::half);
    const size_t packed_k_bytes_per_block = fattn_v2_decode_gqa_slm<64>::K_PACKED_ELEMS * sizeof(sycl::half);
    const int    preferred_tk             = caps.m1n64_k32_supported ? 32 : 16;
    const int    alternate_tk             = preferred_tk == 32 ? 16 : (caps.m1n64_k32_supported ? 32 : 0);
    if (caps.local_mem_size < required_slm_bytes) {
        return { ggml_sycl_fattn_xmx_decode_kv_layout_kind::REJECT,
                 ggml_sycl_fattn_xmx_decode_kv_layout_reason::LOCAL_MEM_UNSUPPORTED,
                 D,
                 params.ne02,
                 params.ne12,
                 n_rep,
                 0,
                 alternate_tk,
                 XMX_V2_DECODE_BATCH_KV,
                 required_slm_bytes,
                 source_k_bytes_per_block,
                 packed_k_bytes_per_block,
                 packed_k_bytes_per_block - source_k_bytes_per_block };
    }

    return { ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE,
             ggml_sycl_fattn_xmx_decode_kv_layout_reason::OK,
             D,
             params.ne02,
             params.ne12,
             n_rep,
             preferred_tk,
             alternate_tk,
             XMX_V2_DECODE_BATCH_KV,
             required_slm_bytes,
             source_k_bytes_per_block,
             packed_k_bytes_per_block,
             packed_k_bytes_per_block - source_k_bytes_per_block };
}

ggml_sycl_fattn_xmx_decode_kv_layout_plan ggml_sycl_flash_attn_ext_xmx_decode_kv_layout_plan(
    const fattn_params & params,
    int                  D,
    const sycl::device & dev,
    bool                 k_device_resident,
    bool                 v_device_resident) {
    ggml_sycl_fattn_xmx_decode_kv_caps caps;
    caps.m1n64_k16_supported = fattn_xmx_v2_decode_m1n64_supported(dev, 16);
    caps.m1n64_k32_supported = fattn_xmx_v2_decode_m1n64_supported(dev, 32);
    caps.local_mem_size      = dev.get_info<sycl::info::device::local_mem_size>();
    caps.k_device_resident   = k_device_resident;
    caps.v_device_resident   = v_device_resident;
    return ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(params, D, caps);
}

bool ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(
    const fattn_params &                                params,
    const ggml_sycl_fattn_xmx_decode_kv_layout_plan &   plan,
    int                                                 target_device,
    ggml_sycl_fattn_xmx_packed_k_materialization_desc * out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};

    if (target_device < 0 || plan.kind != ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE ||
        plan.reason != ggml_sycl_fattn_xmx_decode_kv_layout_reason::OK) {
        return false;
    }
    if (params.K_type != GGML_TYPE_F16 || plan.D != GGML_SYCL_FATTN_XMX_PACKED_K_D ||
        params.ne10 != GGML_SYCL_FATTN_XMX_PACKED_K_D || plan.kv_block_tokens != GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS ||
        plan.packed_k_bytes_per_block != GGML_SYCL_FATTN_XMX_PACKED_K_BYTES_PER_BLOCK) {
        return false;
    }
    if (params.ne11 <= 0 || params.ne12 <= 0 || params.ne13 <= 0 || params.nb11 <= 0 || params.nb12 <= 0 ||
        params.nb13 <= 0) {
        return false;
    }

    const int n_blocks = (params.ne11 + GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS - 1) / GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS;
    const size_t seq_count   = static_cast<size_t>(params.ne13);
    const size_t head_count  = static_cast<size_t>(params.ne12);
    const size_t block_count = static_cast<size_t>(n_blocks);
    if (seq_count > std::numeric_limits<size_t>::max() / head_count) {
        return false;
    }
    const size_t seq_heads = seq_count * head_count;
    if (block_count != 0 && seq_heads > std::numeric_limits<size_t>::max() / block_count) {
        return false;
    }
    const size_t total_blocks = seq_heads * block_count;
    if (total_blocks > std::numeric_limits<size_t>::max() / plan.packed_k_bytes_per_block) {
        return false;
    }

    out->required                    = true;
    out->target_device               = target_device;
    out->D                           = plan.D;
    out->n_kv                        = params.ne11;
    out->H_kv                        = params.ne12;
    out->batch                       = params.ne13;
    out->preferred_tk                = plan.preferred_tk;
    out->alternate_tk                = plan.alternate_tk;
    out->block_tokens                = plan.kv_block_tokens;
    out->n_blocks                    = n_blocks;
    out->source_k_bytes_per_block    = plan.source_k_bytes_per_block;
    out->packed_k_bytes_per_block    = plan.packed_k_bytes_per_block;
    out->packed_k_overhead_per_block = plan.packed_k_overhead_per_block;
    out->total_blocks                = total_blocks;
    out->total_packed_bytes          = total_blocks * plan.packed_k_bytes_per_block;
    out->k_src_nb1                   = params.nb11;
    out->k_src_nb2                   = params.nb12;
    out->k_src_nb3                   = params.nb13;
    out->packed_block_stride         = static_cast<int64_t>(plan.packed_k_bytes_per_block);
    out->packed_head_stride          = static_cast<int64_t>(block_count * plan.packed_k_bytes_per_block);
    out->packed_batch_stride         = static_cast<int64_t>(head_count) * out->packed_head_stride;
    return true;
}

size_t ggml_sycl_fattn_xmx_packed_k_block_offset_bytes(const ggml_sycl_fattn_xmx_packed_k_materialization_desc & desc,
                                                       int sequence,
                                                       int kv_head,
                                                       int block) {
    if (!desc.required || sequence < 0 || sequence >= desc.batch || kv_head < 0 || kv_head >= desc.H_kv || block < 0 ||
        block >= desc.n_blocks) {
        return SIZE_MAX;
    }
    return static_cast<size_t>(sequence) * static_cast<size_t>(desc.packed_batch_stride) +
           static_cast<size_t>(kv_head) * static_cast<size_t>(desc.packed_head_stride) +
           static_cast<size_t>(block) * static_cast<size_t>(desc.packed_block_stride);
}

class fattn_xmx_pack_k_materializer_kernel;

bool ggml_sycl_fattn_xmx_materialize_packed_k(const fattn_params &                              params,
                                              const ggml_sycl_fattn_xmx_decode_kv_layout_plan & plan,
                                              int                                               target_device,
                                              dpct::queue_ptr                                   stream,
                                              ggml_sycl_fattn_xmx_packed_k *                    out) {
    if (stream == nullptr || out == nullptr) {
        return false;
    }

    ggml_sycl_fattn_xmx_packed_k_materialization_desc desc;
    if (!ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(params, plan, target_device, &desc)) {
        return false;
    }
    if (params.K == nullptr) {
        return false;
    }

    const int stream_device = ggml_sycl_get_device_id_from_queue(*stream);
    if (stream_device >= 0 && stream_device != target_device) {
        return false;
    }

    const bool reuse_alloc = out->alloc.ptr != nullptr && out->device == target_device && out->D == desc.D &&
                             out->H_kv == desc.H_kv && out->batch == desc.batch &&
                             out->alloc.tier == ggml_sycl::alloc_tier::DEVICE_VRAM &&
                             out->alloc.size >= desc.total_packed_bytes;
    if (!reuse_alloc) {
        out->reset();

        ggml_sycl::alloc_request req{};
        req.queue                               = stream;
        req.device                              = target_device;
        req.size                                = desc.total_packed_bytes;
        req.intent.role                         = ggml_sycl::alloc_role::KV;
        req.intent.category                     = ggml_sycl::runtime_category::KV_CACHE;
        req.intent.cohort_id                    = "fattn_xmx_packed_k";
        req.intent.constraints.must_device      = true;
        req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::KV;

        if (!ggml_sycl::unified_alloc(req, &out->alloc) || out->alloc.ptr == nullptr ||
            out->alloc.tier != ggml_sycl::alloc_tier::DEVICE_VRAM) {
            out->reset();
            return false;
        }
        out->handle = out->alloc.as_mem_handle();
    }

    sycl::half *      packed_ptr   = static_cast<sycl::half *>(out->alloc.ptr);
    void *            packed_void  = out->alloc.ptr;
    const sycl::event previous_use = out->ready_event;
    bool              add_prev_dep = false;
    if (reuse_alloc) {
        try {
            add_prev_dep = ggml_sycl_should_add_dependency(previous_use);
        } catch (...) {
            add_prev_dep = true;
        }
    }
    const char * K_base     = params.K;
    const size_t total_work = desc.total_blocks * static_cast<size_t>(desc.block_tokens) * static_cast<size_t>(desc.D);

    sycl::event zero_event;
    sycl::event pack_event;
    try {
        zero_event = stream->submit([&](sycl::handler & cgh) {
            if (add_prev_dep) {
                cgh.depends_on(previous_use);
            }
            cgh.memset(packed_void, 0, desc.total_packed_bytes);
        });
        pack_event = stream->submit([&](sycl::handler & cgh) {
            cgh.depends_on(zero_event);
            cgh.parallel_for<fattn_xmx_pack_k_materializer_kernel>(sycl::range<1>(total_work), [=](sycl::id<1> item) {
                const size_t linear   = item[0];
                const int    d        = static_cast<int>(linear % static_cast<size_t>(GGML_SYCL_FATTN_XMX_PACKED_K_D));
                size_t       t        = linear / static_cast<size_t>(GGML_SYCL_FATTN_XMX_PACKED_K_D);
                const int    kv_local = static_cast<int>(t % static_cast<size_t>(GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS));
                t /= static_cast<size_t>(GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS);
                const int block = static_cast<int>(t % static_cast<size_t>(desc.n_blocks));
                t /= static_cast<size_t>(desc.n_blocks);
                const int kv_head  = static_cast<int>(t % static_cast<size_t>(desc.H_kv));
                const int sequence = static_cast<int>(t / static_cast<size_t>(desc.H_kv));

                const int kv_abs = block * GGML_SYCL_FATTN_XMX_PACKED_K_TOKENS + kv_local;
                if (kv_abs >= desc.n_kv) {
                    return;
                }

                const sycl::half k_val = reinterpret_cast<const sycl::half *>(
                    K_base + static_cast<int64_t>(sequence) * desc.k_src_nb3 +
                    static_cast<int64_t>(kv_head) * desc.k_src_nb2 + static_cast<int64_t>(kv_abs) * desc.k_src_nb1)[d];

                const size_t block_base_bytes =
                    static_cast<size_t>(sequence) * static_cast<size_t>(desc.packed_batch_stride) +
                    static_cast<size_t>(kv_head) * static_cast<size_t>(desc.packed_head_stride) +
                    static_cast<size_t>(block) * static_cast<size_t>(desc.packed_block_stride);
                const size_t block_base_half = block_base_bytes / sizeof(sycl::half);
                const size_t elem_off_half   = ggml_sycl_fattn_xmx_packed_k_element_offset_half(kv_local, d);
                packed_ptr[block_base_half + elem_off_half] = k_val;
            });
        });
    } catch (const sycl::exception & e) {
        GGML_LOG_WARN("[SYCL] packed-K materializer submit failed: %s\n", e.what());
        if (!reuse_alloc) {
            out->reset();
        }
        return false;
    }

    out->ready_event = pack_event;
    out->device      = target_device;
    out->D           = desc.D;
    out->n_kv        = desc.n_kv;
    out->H_kv        = desc.H_kv;
    out->batch       = desc.batch;
    out->n_blocks    = desc.n_blocks;
    out->total_bytes = out->alloc.size;
    return true;
}

static void ggml_sycl_fattn_xmx_v2_free_split_workspace(fattn_xmx_v2_device_cache & cache) {
    if (cache.split_partial_max_alloc.ptr) {
        (void) ggml_sycl::unified_free(cache.split_partial_max_alloc);
    }
    if (cache.split_partial_sum_alloc.ptr) {
        (void) ggml_sycl::unified_free(cache.split_partial_sum_alloc);
    }
    if (cache.split_partial_out_alloc.ptr) {
        (void) ggml_sycl::unified_free(cache.split_partial_out_alloc);
    }
    cache.split_partial_max_alloc = {};
    cache.split_partial_sum_alloc = {};
    cache.split_partial_out_alloc = {};
    cache.split_partial_elems     = 0;
    cache.split_partial_out_elems = 0;
    cache.split_partial_device    = -1;
}

static bool ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer(dpct::queue_ptr           stream,
                                                                int                       device,
                                                                size_t                    bytes,
                                                                const char *              cohort_id,
                                                                ggml_sycl::alloc_handle * out) {
    ggml_sycl::alloc_request req{};
    req.queue                               = stream;
    req.device                              = device;
    req.size                                = bytes;
    req.intent.role                         = ggml_sycl::alloc_role::COMPUTE;
    req.intent.category                     = ggml_sycl::runtime_category::COMPUTE;
    req.intent.cohort_id                    = cohort_id;
    req.intent.constraints.must_device      = true;
    req.intent.constraints.prefer_vram_zone = ggml_sycl::vram_zone_id::RUNTIME;
    return ggml_sycl::unified_alloc(req, out) && out->ptr != nullptr && out->tier == ggml_sycl::alloc_tier::DEVICE_VRAM;
}

static bool ggml_sycl_fattn_xmx_v2_ensure_split_workspace(ggml_backend_sycl_context & ctx,
                                                          const fattn_params &        params,
                                                          int                         D,
                                                          int                         n_partitions,
                                                          dpct::queue_ptr             stream,
                                                          float **                    partial_max,
                                                          float **                    partial_sum,
                                                          float **                    partial_out) {
    if (!partial_max || !partial_sum || !partial_out || n_partitions <= 0 || params.ne01 <= 0 || params.ne02 <= 0 ||
        params.ne03 <= 0 || D <= 0) {
        return false;
    }
    size_t partial_elems = 0;
    if ((size_t) params.ne03 > std::numeric_limits<size_t>::max() / (size_t) params.ne02) {
        return false;
    }
    partial_elems = (size_t) params.ne03 * (size_t) params.ne02;
    if (partial_elems > std::numeric_limits<size_t>::max() / (size_t) params.ne01) {
        return false;
    }
    partial_elems *= (size_t) params.ne01;
    if (partial_elems > std::numeric_limits<size_t>::max() / (size_t) n_partitions) {
        return false;
    }
    partial_elems *= (size_t) n_partitions;
    if (partial_elems == 0 || partial_elems > std::numeric_limits<size_t>::max() / (size_t) D) {
        return false;
    }
    const size_t partial_out_elems = partial_elems * (size_t) D;
    if (partial_elems > std::numeric_limits<size_t>::max() / sizeof(float) ||
        partial_out_elems > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }

    fattn_xmx_v2_device_cache * cache = fattn_xmx_v2_get_or_create_cache(ctx);
    const bool                  need_alloc =
        cache->split_partial_device != ctx.device || cache->split_partial_elems < partial_elems ||
        cache->split_partial_out_elems < partial_out_elems || cache->split_partial_max_alloc.ptr == nullptr ||
        cache->split_partial_sum_alloc.ptr == nullptr || cache->split_partial_out_alloc.ptr == nullptr;
    if (need_alloc) {
        if (cache->split_partial_max_alloc.ptr || cache->split_partial_sum_alloc.ptr ||
            cache->split_partial_out_alloc.ptr) {
            try {
                stream->wait_and_throw();
            } catch (const sycl::exception & e) {
                GGML_LOG_WARN("[SYCL] XMX split workspace wait before resize failed: %s\n", e.what());
                return false;
            }
        }
        ggml_sycl_fattn_xmx_v2_free_split_workspace(*cache);

        const size_t partial_bytes     = partial_elems * sizeof(float);
        const size_t partial_out_bytes = partial_out_elems * sizeof(float);
        if (!ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer(
                stream, ctx.device, partial_bytes, "fattn_xmx_split_partial_max", &cache->split_partial_max_alloc) ||
            !ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer(
                stream, ctx.device, partial_bytes, "fattn_xmx_split_partial_sum", &cache->split_partial_sum_alloc) ||
            !ggml_sycl_fattn_xmx_v2_alloc_split_workspace_buffer(stream, ctx.device, partial_out_bytes,
                                                                 "fattn_xmx_split_partial_out",
                                                                 &cache->split_partial_out_alloc)) {
            ggml_sycl_fattn_xmx_v2_free_split_workspace(*cache);
            return false;
        }
        cache->split_partial_elems     = partial_elems;
        cache->split_partial_out_elems = partial_out_elems;
        cache->split_partial_device    = ctx.device;
    }

    *partial_max = static_cast<float *>(cache->split_partial_max_alloc.ptr);
    *partial_sum = static_cast<float *>(cache->split_partial_sum_alloc.ptr);
    *partial_out = static_cast<float *>(cache->split_partial_out_alloc.ptr);
    return true;
}

// ============================================================================
// Multi-sequence boundary detection from mask
// ============================================================================
// In continuous batching with unified KV cache, each query token belongs to a
// sequence, and can only attend to KV positions from its own sequence.
// The mask encodes this: 0 = allow attention, -INF = block attention.
//
// This function scans the F32 mask (before F16 cast) to detect sequence
// boundaries, which allows the kernel to skip cross-sequence computation.
//
// Returns true if sequences were detected, with seq_q_offsets and seq_kv_offsets
// populated. Returns false if single-sequence or detection failed.
//
static bool detect_sequence_boundaries_from_mask(
    const ggml_tensor *    mask_f16,        // The F16 mask tensor (may be a cast of F32)
    int                    device,
    int                    n_queries,       // Number of query tokens
    int                    n_kv,            // Number of KV positions
    std::vector<int32_t> & seq_q_offsets,   // Output: [n_seqs + 1]
    std::vector<int32_t> & seq_kv_offsets)  // Output: [n_seqs + 1]
{
    if (!mask_f16) {
        return false;
    }

    // Try to get the original F32 mask from the cast/copy operation's source
    // ggml_cast uses GGML_OP_CPY with src[0] = source tensor, src[1] = result
    const ggml_tensor * mask_f32 = nullptr;

    if (mask_f16->op == GGML_OP_CPY && mask_f16->src[0] && mask_f16->src[0]->type == GGML_TYPE_F32) {
        mask_f32 = mask_f16->src[0];
    }

    // Need host-accessible F32 mask data
    // NOTE: In SYCL backend, even input tensors are typically in GPU buffers
    // by the time we reach here. Future work: scan mask in kernel using
    // a dedicated work-group, or pass sequence info through ggml graph.
    if (!mask_f32) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(mask_f32->buffer)) {
        return false;
    }

    const float * mask_data = static_cast<const float *>(ggml_sycl_resolve_tensor_ptr(mask_f32, device));
    if (!mask_data) {
        mask_data = static_cast<const float *>(ggml_sycl_host_data(mask_f32));
    }
    if (!mask_data) {
        return false;
    }
    const int64_t mask_stride = mask_f32->nb[1] / sizeof(float);  // Stride between query rows

    // Scan the mask to find sequence boundaries
    // For each query, find the first and last valid (non-INF) KV position
    // Sequence boundaries occur where there's a discontinuity in valid KV ranges

    seq_q_offsets.clear();
    seq_kv_offsets.clear();
    seq_q_offsets.push_back(0);

    int prev_kv_start = -1;
    int prev_kv_end   = -1;

    for (int q = 0; q < n_queries; ++q) {
        const float * row = mask_data + q * mask_stride;

        // Find first valid KV position (value > -1e30)
        int kv_start = -1;
        for (int k = 0; k < n_kv; ++k) {
            if (row[k] > -1e30f) {
                kv_start = k;
                break;
            }
        }

        if (kv_start < 0) {
            // This query has no valid KV positions - shouldn't happen
            // Treat as continuing previous sequence
            continue;
        }

        // Find last valid KV position
        int kv_end = kv_start;
        for (int k = n_kv - 1; k >= kv_start; --k) {
            if (row[k] > -1e30f) {
                kv_end = k + 1;  // End is exclusive
                break;
            }
        }

        // Check if this is a new sequence
        // New sequence if: KV range doesn't overlap with previous, OR
        // KV start is before previous KV start (new sequence started earlier)
        bool new_seq = false;
        if (prev_kv_start < 0) {
            // First query - start first sequence
            new_seq = true;
        } else if (kv_start < prev_kv_start) {
            // KV range starts earlier than previous - definitely new sequence
            new_seq = true;
        } else if (kv_end <= prev_kv_start || kv_start >= prev_kv_end) {
            // Non-overlapping ranges - new sequence
            new_seq = true;
        }

        if (new_seq && q > 0) {
            // End previous sequence, start new one
            seq_q_offsets.push_back(q);
            seq_kv_offsets.push_back(prev_kv_end);
        }

        if (new_seq) {
            // Record this sequence's KV start
            if (seq_kv_offsets.empty()) {
                seq_kv_offsets.push_back(kv_start);
            }
        }

        prev_kv_start = (prev_kv_start < 0) ? kv_start : std::min(prev_kv_start, kv_start);
        prev_kv_end   = std::max(prev_kv_end, kv_end);
    }

    // Close the last sequence
    seq_q_offsets.push_back(n_queries);
    if (prev_kv_end > 0) {
        seq_kv_offsets.push_back(prev_kv_end);
    } else {
        seq_kv_offsets.push_back(n_kv);
    }

    // Only enable multi-sequence optimization if we detected multiple sequences
    int n_seqs = (int) seq_q_offsets.size() - 1;
    if (n_seqs <= 1) {
        seq_q_offsets.clear();
        seq_kv_offsets.clear();
        return false;
    }

    return true;
}

// ============================================================================
// Multi-sequence boundary detection from sequence ID tensors
// ============================================================================
// Computes sequence boundaries from q_seq_ids and kv_seq_ids arrays.
// This enables the kernel to skip cross-sequence KV computation entirely
// (not just mask it), achieving true parallel speedup.
//
// Returns the number of sequences detected (0 if single sequence or error)
//
static int compute_sequence_boundaries_from_ids(const int32_t * q_seq_ids,   // [n_queries] Sequence ID for each query
                                                int             n_queries,
                                                const int32_t * kv_seq_ids,  // [n_kv] Sequence ID for each KV position
                                                int             n_kv,
                                                std::vector<int32_t> & seq_q_offsets,   // Output: [n_seqs + 1]
                                                std::vector<int32_t> & seq_kv_offsets)  // Output: [n_seqs + 1]
{
    if (!q_seq_ids || !kv_seq_ids || n_queries <= 0 || n_kv <= 0) {
        return 0;
    }

    seq_q_offsets.clear();
    seq_kv_offsets.clear();

    // Build a map of sequence_id -> {q_start, q_end, kv_start, kv_end}
    // Assumption: tokens are grouped by sequence (all tokens of seq 0, then seq 1, etc.)
    struct SeqRange {
        int q_start = -1, q_end = -1;
        int kv_start = -1, kv_end = -1;
    };

    std::vector<SeqRange> seq_ranges;

    // Scan query sequence IDs to find sequence boundaries
    int prev_seq = -1;
    for (int q = 0; q < n_queries; ++q) {
        int seq = q_seq_ids[q];
        if (seq != prev_seq) {
            // New sequence starts
            if (seq >= (int) seq_ranges.size()) {
                seq_ranges.resize(seq + 1);
            }
            if (seq_ranges[seq].q_start < 0) {
                seq_ranges[seq].q_start = q;
            }
            prev_seq = seq;
        }
        if (seq >= 0 && seq < (int) seq_ranges.size()) {
            seq_ranges[seq].q_end = q + 1;
        }
    }

    // Scan KV sequence IDs to find KV boundaries
    prev_seq = -1;
    for (int k = 0; k < n_kv; ++k) {
        int seq = kv_seq_ids[k];
        if (seq >= 0 && seq < (int) seq_ranges.size()) {
            if (seq_ranges[seq].kv_start < 0) {
                seq_ranges[seq].kv_start = k;
            }
            seq_ranges[seq].kv_end = k + 1;
        }
    }

    // Build ordered offset arrays for sequences that have both Q and KV
    int n_valid_seqs = 0;
    for (size_t s = 0; s < seq_ranges.size(); ++s) {
        if (seq_ranges[s].q_start >= 0 && seq_ranges[s].kv_start >= 0) {
            if (n_valid_seqs == 0) {
                // First sequence
                seq_q_offsets.push_back(seq_ranges[s].q_start);
                seq_kv_offsets.push_back(seq_ranges[s].kv_start);
            }
            n_valid_seqs++;
            seq_q_offsets.push_back(seq_ranges[s].q_end);
            seq_kv_offsets.push_back(seq_ranges[s].kv_end);
        }
    }

    // Only enable multi-sequence optimization if we detected multiple sequences
    if (n_valid_seqs <= 1) {
        seq_q_offsets.clear();
        seq_kv_offsets.clear();
        return 0;
    }

    return n_valid_seqs;
}

// Kernel selection is now done at runtime based on GPU capabilities.
// XMX kernel (3) is used on Intel GPUs with matrix extension support (Arc, etc.)
// MMA F16 kernel (2) is used on other SYCL devices.
// Kernel IDs:
// 0 = VEC kernel (simpler, one K/V position at a time)
// 1 = MMA kernel (tiled scalar, processes BATCH_KV positions at a time)
// 2 = MMA F16 kernel (scalar with SG_SIZE=16, named MMA but not using joint_matrix)
// 3 = XMX F16 kernel (using joint_matrix for Q@K^T acceleration)

// Check if flash attention is supported for the given operation
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    // Enabled by default; allow explicit disable for debugging/regressions.
    static const bool fa_ext_enabled = []() {
        const char * env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");
        if (!env) {
            return true;
        }
        return !(strcmp(env, "0") == 0 || strcmp(env, "false") == 0);
    }();
    if (!fa_ext_enabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Check Q type - must be F32 or F16
    if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {
        return false;
    }

    // Check K/V types - F16 or FP8 E4M3 (for memory savings)
    const bool kv_is_fp8 = (ggml_sycl_type_is_fp8_e4m3(K->type) && ggml_sycl_type_is_fp8_e4m3(V->type));
    const bool kv_is_f16 = (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    if (!kv_is_f16 && !kv_is_fp8) {
        return false;
    }

    // Check destination type
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Masked flash attention is supported; keep the gate in place via GGML_SYCL_FLASH_ATTN_EXT if needed.

    // Check head dimension - must be a supported size
    const int D = Q->ne[0];
    if (!fattn_vec_supports_head_dim(D)) {
        return false;
    }

    // MMA kernel handles high head counts (>32), vec kernel handles <=32
    // Both are now supported

    // Check that tensors are contiguous
    if (Q->nb[0] != ggml_type_size(Q->type)) {
        return false;
    }
    // K/V stride depends on type: 2 bytes for F16, 1 byte for FP8
    const size_t kv_elem_size = kv_is_fp8 ? sizeof(uint8_t) : sizeof(sycl::half);
    if (K->nb[0] != kv_elem_size) {
        return false;
    }
    if (V->nb[0] != kv_elem_size) {
        return false;
    }
    if (mask && mask->nb[0] != sizeof(sycl::half)) {
        return false;
    }

    return true;
}

// =============================================================================
// Shape-aware FA dispatch table (5edo8.5)
// =============================================================================

// Dispatcher that selects appropriate kernel based on head dimension and GPU capabilities
template <int D, typename Q_type>
static void ggml_sycl_flash_attn_ext_dispatch_ncols(ggml_backend_sycl_context & ctx, const fattn_params & params) {
    dpct::queue_ptr stream = ctx.stream();

    // Select ncols based on batch size (ne01 = number of queries)
    const int ne01          = params.ne01;
    float     logit_softcap = params.logit_softcap;

    // Runtime kernel selection based on GPU capabilities
    // Check if the device has XMX (Intel matrix extension) support
    sycl::device dev                = stream->get_device();
    bool         use_xmx            = gpu_has_xmx(dev);
    const auto   fast_decode_policy = ggml_sycl_fattn_fast_decode_policy(params, D);
    const bool   fast_esimd_decode =
        fast_decode_policy.fast_esimd_safe && g_sycl_fa_esimd_enabled && fattn_esimd_f16_available();
    const bool safe_decode =
        g_sycl_fa_safe_decode_enabled && ne01 <= 1 && (g_sycl_fa_safe_decode_forced || !fast_esimd_decode);
    if (safe_decode) {
        use_xmx = false;
    }
    {
        static const bool force_no_xmx = []() {
            const char * env = std::getenv("GGML_SYCL_FA_NO_XMX");
            return env && std::atoi(env) != 0;
        }();
        if (force_no_xmx) {
            use_xmx = false;
        }
    }

    // Helper macro to dispatch softcap — defined here so FA_FORCE_PATH can use it.
    // v2 variant: dispatcher takes `ctx` first and returns bool (XMX-v2 pattern).
#define DISPATCH_NCOLS(NCOLS, LAUNCHER)                    \
    if (logit_softcap == 0.0f) {                           \
        LAUNCHER<D, NCOLS, false, Q_type>(params, stream); \
    } else {                                               \
        LAUNCHER<D, NCOLS, true, Q_type>(params, stream);  \
    }

#define DISPATCH_NCOLS_BATCH_KV(NCOLS, BATCH_KV, LAUNCHER)           \
    if (logit_softcap == 0.0f) {                                     \
        LAUNCHER<D, NCOLS, false, Q_type, BATCH_KV>(params, stream); \
    } else {                                                         \
        LAUNCHER<D, NCOLS, true, Q_type, BATCH_KV>(params, stream);  \
    }

// Same but launcher takes `ctx` first and returns bool (XMX-v2 pattern).
#define DISPATCH_NCOLS_CTX_ACC(NCOLS, LAUNCHER, ACC_T)                                 \
    if (logit_softcap == 0.0f) {                                                       \
        v2_dispatched = LAUNCHER<D, NCOLS, false, Q_type, ACC_T>(ctx, params, stream); \
    } else {                                                                           \
        v2_dispatched = LAUNCHER<D, NCOLS, true, Q_type, ACC_T>(ctx, params, stream);  \
    }

#define DISPATCH_NCOLS_CTX_ACC_VARIANT(NCOLS, LAUNCHER, ACC_T, VARIANT)                         \
    if (logit_softcap == 0.0f) {                                                                \
        v2_dispatched = LAUNCHER<D, NCOLS, false, Q_type, ACC_T>(ctx, params, stream, VARIANT); \
    } else {                                                                                    \
        v2_dispatched = LAUNCHER<D, NCOLS, true, Q_type, ACC_T>(ctx, params, stream, VARIANT);  \
    }

    // ---- Debug trace: gated by GGML_SYCL_FA_DISPATCH_DEBUG, limited by GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT ----
    static std::atomic<int> fattn_dispatch_debug_count = 0;
    int        dispatch_debug_counter = fattn_dispatch_debug_count.fetch_add(1, std::memory_order_relaxed) + 1;
    int        debug_limit            = std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT") ?
                                            std::atoi(std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG_LIMIT")) :
                                            32;
    const bool dispatch_debug_enabled =
        std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG") && dispatch_debug_counter <= debug_limit;
    auto dispatch_debug_kernel = [&](const char * kernel) {
        if (dispatch_debug_enabled) {
            fprintf(stderr, "[SYCL] fattn selected [%d/%d] %s D=%d ne01=%d safe_decode=%d fast_esimd_safe=%d\n",
                    dispatch_debug_counter, debug_limit, kernel, D, ne01, (int) safe_decode,
                    (int) fast_decode_policy.fast_esimd_safe);
        }
    };
    if (dispatch_debug_enabled) {
        const char * selected = safe_decode ? "(safe)" :
                                ne01 <= 1   ? "ne1=1" :
                                ne01 <= 2   ? "ne1<=2" :
                                ne01 <= 4   ? "ne1<=4" :
                                ne01 <= 16  ? "ne1<=16" :
                                              "ne1>16";
        fprintf(stderr,
                "[SYCL] fattn dispatch [%d/%d] D=%d %s xmx=%d safe_decode=%d fast_esimd_safe=%d forced_safe=%d\n",
                dispatch_debug_counter, debug_limit, D, selected, (int) use_xmx, (int) safe_decode,
                (int) fast_decode_policy.fast_esimd_safe, (int) g_sycl_fa_safe_decode_forced);
    }

    auto can_use_esimd_batched_pp = [&](bool ignore_safe_decode) -> bool {
        if constexpr (D != 64) {
            return false;
        } else {
            if (!ignore_safe_decode && safe_decode) {
                return false;
            }
            if (ne01 <= 1 || !g_sycl_fa_esimd_enabled || !fattn_esimd_f16_available()) {
                return false;
            }
            if (params.kv_is_fp8 || params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
                return false;
            }
            if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
                return false;
            }
            if (params.mask != nullptr && params.mask_type != GGML_TYPE_F16) {
                return false;
            }
            if (params.max_bias != 0.0f) {
                return false;
            }
            if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
                params.seq_lens != nullptr) {
                return false;
            }
            if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
                params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
                return false;
            }
            if (params.multi_token_decode || params.q_positions != nullptr) {
                return false;
            }
            if (params.ne12 <= 0 || params.ne02 <= 0 || params.ne02 % params.ne12 != 0) {
                return false;
            }

            const size_t local_mem_size = (size_t) dev.get_info<sycl::info::device::local_mem_size>();
            return local_mem_size >= fattn_esimd_batched_slm_bytes<D>();
        }
    };

    auto can_use_xmx_v2_d64_batched_pp = [&]() -> bool {
        if constexpr (D != 64) {
            return false;
        } else {
            // XMX-v2 is currently proven for real PP batches. Tiny multi-query
            // batches stay on ESIMD until separately measured.
            if (!use_xmx || safe_decode || ne01 < 64 || !fattn_xmx_v2_f16_available()) {
                return false;
            }
            if (params.kv_is_fp8 || params.K_type != GGML_TYPE_F16 || params.V_type != GGML_TYPE_F16) {
                return false;
            }
            if (params.Q_type != GGML_TYPE_F16 && params.Q_type != GGML_TYPE_F32) {
                return false;
            }
            if (params.mask != nullptr && params.mask_type != GGML_TYPE_F16) {
                return false;
            }
            if (params.max_bias != 0.0f) {
                return false;
            }
            if (params.use_paged_attn || params.use_paged_layout || params.block_table != nullptr ||
                params.seq_lens != nullptr) {
                return false;
            }
            if (params.n_seqs > 1 || params.seq_q_offsets != nullptr || params.seq_kv_offsets != nullptr ||
                params.q_seq_ids != nullptr || params.kv_seq_ids != nullptr) {
                return false;
            }
            if (params.multi_token_decode || params.q_positions != nullptr) {
                return false;
            }
            return params.ne12 > 0 && params.ne02 > 0 && params.ne02 % params.ne12 == 0;
        }
    };

    auto can_use_xmx_v1_runtime = [&]() -> bool {
        // XMX-v1 is the old diagnostic kernel.  It still has measurable drift
        // on GPT-OSS sink+softcap shapes and D128 ragged query tails, so only
        // allow it for the simple full-8-query D=128 PP shape family it was
        // historically used for. Everything else must use v2 or a later proven
        // path.
        if constexpr (D != 128) {
            return false;
        } else {
            return ne01 >= 8 && ne01 % 8 == 0 && params.sinks == nullptr && params.logit_softcap == 0.0f &&
                   !params.kv_is_fp8 && !params.use_paged_attn && !params.use_paged_layout &&
                   params.block_table == nullptr && params.seq_lens == nullptr && params.n_seqs <= 1 &&
                   params.seq_q_offsets == nullptr && params.seq_kv_offsets == nullptr && params.q_seq_ids == nullptr &&
                   params.kv_seq_ids == nullptr && !params.multi_token_decode && params.q_positions == nullptr;
        }
    };

#if GGML_SYCL_DNNL
    auto onednn_layout_reason_name = [](ggml_sycl_onednn_fa_layout_reason reason) {
        switch (reason) {
            case ggml_sycl_onednn_fa_layout_reason::OK:
                return "ok";
            case ggml_sycl_onednn_fa_layout_reason::BELOW_MIN_NCOLS:
                return "below-min-ncols";
            case ggml_sycl_onednn_fa_layout_reason::SINKS_UNSUPPORTED:
                return "sinks-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::SOFTCAP_UNSUPPORTED:
                return "softcap-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::MAX_BIAS_UNSUPPORTED:
                return "max-bias-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::FP8_KV_UNSUPPORTED:
                return "fp8-kv-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::MULTI_SEQ_UNSUPPORTED:
                return "multi-seq-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::BATCH_UNSUPPORTED:
                return "batch-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::PAGED_UNSUPPORTED:
                return "paged-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D:
                return "unsupported-D";
            case ggml_sycl_onednn_fa_layout_reason::EMPTY_KV:
                return "empty-kv";
            case ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH:
                return "kv-nc-stride-mismatch";
            case ggml_sycl_onednn_fa_layout_reason::HEAD_RATIO_UNSUPPORTED:
                return "head-ratio-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::Q_TYPE_UNSUPPORTED:
                return "q-type-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::K_TYPE_UNSUPPORTED:
                return "k-type-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::V_TYPE_UNSUPPORTED:
                return "v-type-unsupported";
            case ggml_sycl_onednn_fa_layout_reason::MASK_TYPE_UNSUPPORTED:
                return "mask-type-unsupported";
        }
        return "unknown";
    };
#endif

    // ---- GGML_SYCL_FA_FORCE_PATH: override all shape-based kernel selection ----
    {
        const char * force = std::getenv("GGML_SYCL_FA_FORCE_PATH");
        if (force) {
            bool force_known = false;

            if (strcmp(force, "vec") == 0) {
                force_known = true;
                if (!params.kv_is_fp8 && ne01 <= 1 && D <= 256) {
                    GGML_SYCL_KTRACE("fattn FORCE=vec", " D=%d ncols=1 ne01=%d", D, ne01);
                    dispatch_debug_kernel("force_vec_f16");
                    DISPATCH_NCOLS(1, launch_fattn_vec_f16);
                    return;
                }
                if (dispatch_debug_enabled) {
                    fprintf(stderr,
                            "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=vec deferred; requires f16/f32 single-query "
                            "decode with D<=256 (ne01=%d D=%d kv_fp8=%d)\n",
                            ne01, D, (int) params.kv_is_fp8);
                }
            }
#if GGML_SYCL_DNNL
            if (strcmp(force, "onednn") == 0 && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {
                force_known                                     = true;
                const bool                            multi_seq = (params.n_seqs > 1);
                const ggml_sycl_onednn_fa_layout_plan plan =
                    ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8, multi_seq);
                if (plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT ||
                    plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED) {
                    GGML_SYCL_KTRACE("fattn FORCE=onednn", " D=%d ne01=%d ne11=%d materialize=%d", D, ne01, params.ne11,
                                     (int) (plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED));
                    if (ggml_sycl_flash_attn_ext_onednn(ctx, params)) {
                        dispatch_debug_kernel("force_onednn");
                        return;
                    }
                    if (plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED) {
                        fprintf(stderr,
                                "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=onednn MATERIALIZE_REQUIRED_BUT_UNAVAILABLE "
                                "reason=%s\n",
                                onednn_layout_reason_name(plan.reason));
                    }
                } else {
                    fprintf(
                        stderr,
                        "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=onednn REJECTED unsupported layout: kind=%d reason=%s\n",
                        (int) plan.kind, onednn_layout_reason_name(plan.reason));
                }
            }
#endif
            if (strcmp(force, "xmx-v1") == 0) {
                // v1 kernel — kept for A/B comparison only — use GGML_SYCL_FA_XMX_V1 instead
                force_known = true;
                if (can_use_xmx_v1_runtime()) {
                    GGML_SYCL_KTRACE("fattn FORCE=xmx-v1", " D=%d ncols=%d ne01=%d", D, 8, ne01);
                    dispatch_debug_kernel("force_xmx_v1_f16");
                    DISPATCH_NCOLS(8, launch_fattn_xmx_f16);
                    return;
                }
                if (dispatch_debug_enabled) {
                    fprintf(stderr,
                            "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=xmx-v1 rejected for D=%d ne01=%d sinks=%d "
                            "softcap=%.6g fp8=%d; falling through\n",
                            D, ne01, params.sinks != nullptr, params.logit_softcap, (int) params.kv_is_fp8);
                }
            }
            if (strcmp(force, "xmx-v2-decode") == 0 || strcmp(force, "xmx-v2-m1n64") == 0) {
                force_known      = true;
                bool decode_sent = false;
                if constexpr (D == 64) {
                    if (ne01 <= 1 && !safe_decode) {
                        GGML_SYCL_KTRACE("fattn FORCE=xmx-v2-decode", " D=%d ne01=%d", D, ne01);
                        if (logit_softcap == 0.0f) {
                            decode_sent = launch_fattn_xmx_v2_decode_m1n64<D, false, Q_type>(ctx, params, stream);
                        } else {
                            decode_sent = launch_fattn_xmx_v2_decode_m1n64<D, true, Q_type>(ctx, params, stream);
                        }
                    }
                }
                dispatch_debug_kernel(decode_sent ? "force_xmx_v2_decode_m1n64" : "force_xmx_v2_decode_m1n64_rejected");
                if (decode_sent) {
                    return;
                }
                if (ne01 <= 1) {
                    DISPATCH_NCOLS(1, launch_fattn_tile_f16);
                    return;
                }
            }
            if (strcmp(force, "xmx-v2-decode-gqa") == 0 || strcmp(force, "xmx-v2-gqa") == 0) {
                force_known      = true;
                bool decode_sent = false;
                if constexpr (D == 64) {
                    if (ne01 <= 1 && !safe_decode) {
                        GGML_SYCL_KTRACE("fattn FORCE=xmx-v2-decode-gqa", " D=%d ne01=%d", D, ne01);
                        if (logit_softcap == 0.0f) {
                            decode_sent = launch_fattn_xmx_v2_decode_gqa<D, false, Q_type>(ctx, params, stream);
                        } else {
                            decode_sent = launch_fattn_xmx_v2_decode_gqa<D, true, Q_type>(ctx, params, stream);
                        }
                    }
                }
                dispatch_debug_kernel(decode_sent ? "force_xmx_v2_decode_gqa" : "force_xmx_v2_decode_gqa_rejected");
                if (decode_sent) {
                    return;
                }
                if (ne01 <= 1) {
                    DISPATCH_NCOLS(1, launch_fattn_tile_f16);
                    return;
                }
            }
            if (strcmp(force, "xmx-v2-decode-gqa-split-packed") == 0 ||
                strcmp(force, "xmx-v2-decode-gqa-split-packed-k32") == 0 ||
                strcmp(force, "xmx-v2-decode-gqa-split-packed-k16") == 0 ||
                strcmp(force, "xmx-v2-gqa-split-packed") == 0 || strcmp(force, "xmx-v2-gqa-split-packed-k32") == 0 ||
                strcmp(force, "xmx-v2-gqa-split-packed-k16") == 0) {
                force_known      = true;
                bool decode_sent = false;
                if constexpr (D == 64) {
                    if (ne01 <= 1 && !safe_decode) {
                        const bool force_tk16 = strstr(force, "k16") != nullptr;
                        const bool force_tk32 = strstr(force, "k32") != nullptr;
                        const bool k_device_resident =
                            ggml_sycl_is_device_ptr(params.K) ||
                            ggml_sycl_probe_alloc_type_any_context(params.K) == sycl::usm::alloc::device;
                        const bool v_device_resident =
                            ggml_sycl_is_device_ptr(params.V) ||
                            ggml_sycl_probe_alloc_type_any_context(params.V) == sycl::usm::alloc::device;
                        const ggml_sycl_fattn_xmx_decode_kv_layout_plan plan =
                            ggml_sycl_flash_attn_ext_xmx_decode_kv_layout_plan(params, D, dev, k_device_resident,
                                                                                       v_device_resident);
                        if (plan.kind == ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE) {
                            fattn_xmx_v2_device_cache * cache = fattn_xmx_v2_get_or_create_cache(ctx);
                            const int                   n_partitions =
                                (params.ne11 + XMX_V2_DECODE_BATCH_KV - 1) / XMX_V2_DECODE_BATCH_KV;
                            const int selected_tk = force_tk16 ? 16 : force_tk32 ? 32 : plan.preferred_tk;
                            float * partial_max = nullptr;
                            float * partial_sum = nullptr;
                            float * partial_out = nullptr;
                            ggml_sycl_fattn_xmx_packed_k * packed_k =
                                ggml_sycl_fattn_xmx_find_packed_k_sidecar(params, ctx.device);
                            if (packed_k == nullptr &&
                                ggml_sycl_fattn_xmx_materialize_packed_k(params, plan, ctx.device, stream,
                                                                         &cache->forced_split_packed_k)) {
                                packed_k = &cache->forced_split_packed_k;
                            }
                            if (packed_k != nullptr &&
                                ggml_sycl_fattn_xmx_v2_ensure_split_workspace(
                                    ctx, params, D, n_partitions, stream, &partial_max, &partial_sum, &partial_out)) {
                                const bool used_sidecar = packed_k != &cache->forced_split_packed_k;
                                if (dispatch_debug_enabled) {
                                    fprintf(stderr,
                                            "[SYCL] fattn packed-K force path: tk=%d sidecar=%d k_view_offs=%zu "
                                            "n_partitions=%d\n",
                                            selected_tk, (int) used_sidecar, params.K_view_offs, n_partitions);
                                }
                                GGML_SYCL_KTRACE("fattn FORCE=xmx-v2-gqa-split-packed", " D=%d ne01=%d tk=%d sidecar=%d",
                                                 D, ne01, selected_tk, used_sidecar ? 1 : 0);
                                if (logit_softcap == 0.0f) {
                                    decode_sent =
                                        selected_tk == 16 ?
                                            launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, Q_type, 16>(
                                                ctx, params, stream, packed_k, partial_max, partial_sum, partial_out,
                                                n_partitions) :
                                            launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, false, Q_type, 32>(
                                                ctx, params, stream, packed_k, partial_max, partial_sum, partial_out,
                                                n_partitions);
                                } else {
                                    decode_sent =
                                        selected_tk == 16 ?
                                            launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, true, Q_type, 16>(
                                                ctx, params, stream, packed_k, partial_max, partial_sum, partial_out,
                                                n_partitions) :
                                            launch_fattn_xmx_v2_decode_gqa_split_packed_tk<D, true, Q_type, 32>(
                                                ctx, params, stream, packed_k, partial_max, partial_sum, partial_out,
                                                n_partitions);
                                }
                            }
                        } else if (dispatch_debug_enabled) {
                            fprintf(stderr,
                                    "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=%s rejected packed-K plan kind=%d reason=%d "
                                    "k_device=%d v_device=%d\n",
                                    force, (int) plan.kind, (int) plan.reason, (int) k_device_resident,
                                    (int) v_device_resident);
                        }
                    }
                }
                dispatch_debug_kernel(decode_sent ? "force_xmx_v2_decode_gqa_split_packed" :
                                                    "force_xmx_v2_decode_gqa_split_packed_rejected");
                if (decode_sent) {
                    return;
                }
                if (ne01 <= 1) {
                    DISPATCH_NCOLS(1, launch_fattn_tile_f16);
                    return;
                }
            }
            if (strcmp(force, "xmx-v2") == 0 || strcmp(force, "xmx-v2-tm16") == 0 || strcmp(force, "xmx-v2-tm8") == 0) {
                force_known               = true;
                const int  forced_variant = strcmp(force, "xmx-v2-tm16") == 0 ? 1 :
                                            strcmp(force, "xmx-v2-tm8") == 0  ? 0 :
                                                                                -1;
                const bool use_f32_acc    = (params.prec == GGML_PREC_F32 || params.max_bias != 0.0f);
                bool       v2_dispatched  = false;

                if (ne01 <= 1) {
                    GGML_SYCL_KTRACE("fattn FORCE=xmx-v2", " D=%d ncols=1 ne01=%d variant=%d", D, ne01, forced_variant);
                    if (use_f32_acc) {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(1, launch_fattn_xmx_v2_f16_variant, float, forced_variant);
                    } else {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(1, launch_fattn_xmx_v2_f16_variant, afloat, forced_variant);
                    }
                    dispatch_debug_kernel(v2_dispatched ? "force_xmx_v2_f16_ncols1" :
                                                          "force_xmx_v2_f16_ncols1_rejected");
                    if (!v2_dispatched) {
                        DISPATCH_NCOLS(1, launch_fattn_tile_f16);
                    }
                } else if (ne01 <= 2) {
                    GGML_SYCL_KTRACE("fattn FORCE=xmx-v2", " D=%d ncols=2 ne01=%d variant=%d", D, ne01, forced_variant);
                    if (use_f32_acc) {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(2, launch_fattn_xmx_v2_f16_variant, float, forced_variant);
                    } else {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(2, launch_fattn_xmx_v2_f16_variant, afloat, forced_variant);
                    }
                    dispatch_debug_kernel(v2_dispatched ? "force_xmx_v2_f16_ncols2" :
                                                          "force_xmx_v2_f16_ncols2_rejected");
                    if (!v2_dispatched) {
                        DISPATCH_NCOLS(2, launch_fattn_tile_f16);
                    }
                } else if (ne01 <= 4) {
                    GGML_SYCL_KTRACE("fattn FORCE=xmx-v2", " D=%d ncols=8 ne01=%d variant=%d", D, ne01, forced_variant);
                    if (use_f32_acc) {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(8, launch_fattn_xmx_v2_f16_variant, float, forced_variant);
                    } else {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(8, launch_fattn_xmx_v2_f16_variant, afloat, forced_variant);
                    }
                    dispatch_debug_kernel(v2_dispatched ? "force_xmx_v2_f16_ncols8" :
                                                          "force_xmx_v2_f16_ncols8_rejected");
                    if (!v2_dispatched) {
                        DISPATCH_NCOLS(4, launch_fattn_tile_f16);
                    }
                } else {
                    GGML_SYCL_KTRACE("fattn FORCE=xmx-v2", " D=%d ncols=16 ne01=%d variant=%d", D, ne01,
                                     forced_variant);
                    if (use_f32_acc) {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(16, launch_fattn_xmx_v2_f16_variant, float, forced_variant);
                    } else {
                        DISPATCH_NCOLS_CTX_ACC_VARIANT(16, launch_fattn_xmx_v2_f16_variant, afloat, forced_variant);
                    }
                    dispatch_debug_kernel(v2_dispatched ? "force_xmx_v2_f16_ncols16" :
                                                          "force_xmx_v2_f16_ncols16_rejected");
                    if (!v2_dispatched) {
                        DISPATCH_NCOLS(8, launch_fattn_tile_f16);
                    }
                }
                return;
            }
            if (strcmp(force, "esimd") == 0) {
                force_known = true;
                if (g_sycl_fa_esimd_enabled && fattn_esimd_f16_available()) {
                    if (ne01 <= 1) {
                        GGML_SYCL_KTRACE("fattn FORCE=esimd", " D=%d ne01=%d", D, ne01);
                        dispatch_debug_kernel("force_esimd_f16");
                        fattn_esimd_f16<D, Q_type>(params, *stream);
                        return;
                    }
                    if (can_use_esimd_batched_pp(true)) {
                        GGML_SYCL_KTRACE("fattn FORCE=esimd_batched", " D=%d ne01=%d", D, ne01);
                        dispatch_debug_kernel("force_esimd_f16_batched");
                        fattn_esimd_f16_batched<D, Q_type>(params, *stream);
                        return;
                    }
                }
                if (dispatch_debug_enabled) {
                    fprintf(stderr,
                            "[SYCL] fattn: GGML_SYCL_FA_FORCE_PATH=esimd unavailable for D=%d ne01=%d; falling "
                            "through\n",
                            D, ne01);
                }
            }
            if (strcmp(force, "tile") == 0) {
                force_known = true;
                GGML_SYCL_KTRACE("fattn FORCE=tile", " D=%d ncols=%d ne01=%d", D, 8, ne01);
                dispatch_debug_kernel("force_tile_f16");
                DISPATCH_NCOLS(8, launch_fattn_tile_f16);
                return;
            }
            if (!force_known) {
                fprintf(stderr, "[SYCL] fattn: unknown GGML_SYCL_FA_FORCE_PATH=%s, falling through\n", force);
            }
        }
    }

    // ---- ESIMD decode dispatch (uses explicit SIMD operations) ----
    // Single query (ne01 <= 1): +7% speedup for decode on Mistral 7B.
    // Batched D=64 PP is selected below after oneDNN has had first refusal.
    if (!safe_decode && fast_esimd_decode) {
        if (ne01 <= 1) {
            // Partitioned kernel: each thread processes disjoint KV ranges
            GGML_SYCL_KTRACE("fattn_esimd_f16", " D=%d ne01=%d", D, ne01);
            dispatch_debug_kernel("esimd_f16");
            fattn_esimd_f16<D, Q_type>(params, *stream);
            return;
        }
    }

    // VEC path: deterministic-by-construction TG fast path (zero SLM, register-only).
    // ncols == 1 covers TG; D <= 256 covers gpt-oss-20b (D=64), Mistral/Llama (D=128), Gemma (D=256).
    // This path IS the deterministic default for TG — safe_decode uses the scalar MMA fallback
    // only when GGML_SYCL_FA_SAFE_DECODE is explicitly left at default (true).
    // VEC replaces BOTH the racy XMX (ne01=1 wastes 7/8 tile) and the safe_decode scalar default.
    if (!params.kv_is_fp8 && ne01 <= 1 && D <= 256) {
        GGML_SYCL_KTRACE("fattn_vec_f16", " D=%d ncols=1 ne01=%d", D, ne01);
        dispatch_debug_kernel("vec_f16");
        DISPATCH_NCOLS(1, launch_fattn_vec_f16);
        return;
    }

#if GGML_SYCL_DNNL
    // oneDNN graph SDPA path: fused MatMul→Divide→Add→SoftMax→MatMul on Xe2.
    // Retired by default for Mistral-like GQA shapes (nc≠D) since oneDNN's
    // 4-D strided layout cannot represent non-contiguous nc-D K planes.
    // Eligible for non-GQA and nc==D shapes.
    // GGML_SYCL_FA_ONEDNN_ALLOW intentionally does not bypass the nc==D gate:
    // Mistral-like GQA layouts produced deterministic completion corruption.
    // Other eligibility: no sinks, no softcap, f16 KV, D ≤ 512, not multi-
    // seq, not paged-v2, not safe_decode. The paged-v2 block layout stores
    // K/V as [D, block_size, n_blocks] rather than the contiguous [D, n_kv]
    // that the oneDNN graph expects. Paged-v2 keeps its own dispatch path
    // below and must win before the oneDNN branch considers the op.
    // Ineligible: gpt-oss-20b (sinks+softcap), Gemma-2 (softcap), FP8 KV.
    // Falls back to kernel path if oneDNN compile fails or during graph recording.
    if (!safe_decode && g_sycl_fa_onednn_enabled && !g_sycl_paged_v2_enabled) {
        const bool                            multi_seq = (params.n_seqs > 1);
        const ggml_sycl_onednn_fa_layout_plan plan      = ggml_sycl_flash_attn_ext_onednn_plan(params,
                                                                                               params.ne02,  // H_q
                                                                                               params.ne12,  // H_kv
                                                                                               params.kv_is_fp8, multi_seq);
        if (plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT ||
            plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED) {
            GGML_SYCL_KTRACE("fattn_onednn", " D=%d ne01=%d ne11=%d H_q=%d H_kv=%d materialize=%d", D, ne01,
                             params.ne11, params.ne02, params.ne12,
                             (int) (plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED));
            if (ggml_sycl_flash_attn_ext_onednn(ctx, params)) {
                dispatch_debug_kernel("onednn");
                return;
            }
            if (plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED &&
                std::getenv("GGML_SYCL_FA_DISPATCH_DEBUG") && dispatch_debug_counter <= debug_limit) {
                fprintf(
                    stderr,
                    "[SYCL] fattn: oneDNN MATERIALIZE_REQUIRED_BUT_UNAVAILABLE; falling back to native FA: reason=%s\n",
                    onednn_layout_reason_name(plan.reason));
            }
        }
    }
#endif  // GGML_SYCL_DNNL

    // Batched D=64 PP: XMX-v2 is the default when the shape/capability guards
    // match. B50 GPT-OSS profiling in llama.cpp-v90xn.32.6 measured the v2
    // kernel ahead of ESIMD for PP, while decode/TG stayed faster on ESIMD/VEC.
    // The v2 launcher still owns the per-device matrix/SLM fit check; rejection
    // falls through to ESIMD without requiring an environment override.
    if (can_use_xmx_v2_d64_batched_pp()) {
        bool       v2_dispatched = false;
        // D=64 batched PP is bandwidth/occupancy limited on Xe.  Microbenchmarks
        // for GPT-OSS-like sinks/softcap shapes show the build-time afloat
        // accumulator keeps public-layout error below 2e-4.  NCOLS=32 reuses
        // each K/V tile across a wider query block and cuts pp512+sinks+softcap
        // latency from ~7.8 ms/layer (f32,ncols16) to ~1.9 ms/layer.
        GGML_SYCL_KTRACE("fattn_xmx_v2_f16_pp", " D=%d ncols=32 ne01=%d prec=%d acc=afloat", D, ne01,
                         (int) params.prec);
        DISPATCH_NCOLS_CTX_ACC(32, launch_fattn_xmx_v2_f16, afloat);
        dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_pp_ncols32" : "xmx_v2_f16_pp_ncols32_rejected");
        if (v2_dispatched) {
            return;
        }
    }

    // Batched ESIMD D=64 PP fallback. This remains the proven fallback for
    // devices without XMX-v2 support or shapes rejected by the v2 launcher.
    if (can_use_esimd_batched_pp(false)) {
        GGML_SYCL_KTRACE("fattn_esimd_f16_batched", " D=%d ne01=%d slm=%zu", D, ne01,
                         fattn_esimd_batched_slm_bytes<D>());
        dispatch_debug_kernel("esimd_f16_batched");
        fattn_esimd_f16_batched<D, Q_type>(params, *stream);
        return;
    }

    // Dispatch remaining shapes based on GPU capabilities.
    // XMX-v2 (fattn-xmx-f16-v2.hpp) has no SLM aliasing and remains the
    // XMX-capable path for shapes not claimed by the proven fast paths above.
    // GGML_SYCL_FA_XMX_V1=1: fallback to broken v1 kernel for A/B comparison only.
    if (use_xmx) {
        static const bool force_xmx_v1 = []() {
            const char * env = std::getenv("GGML_SYCL_FA_XMX_V1");
            return env && std::atoi(env) != 0;
        }();
        static const bool disable_xmx_v1_pp = []() {
            const char * env = std::getenv("GGML_SYCL_FA_XMX_V1_PP");
            return env && std::atoi(env) == 0;
        }();
        const bool xmx_v1_supported = can_use_xmx_v1_runtime();
        const bool simple_pp_xmx_v1 = !disable_xmx_v1_pp && xmx_v1_supported;
        const bool use_xmx_v1_path  = xmx_v1_supported && (force_xmx_v1 || simple_pp_xmx_v1);
        if (force_xmx_v1 && !xmx_v1_supported && dispatch_debug_enabled) {
            fprintf(stderr,
                    "[SYCL] fattn: GGML_SYCL_FA_XMX_V1=1 rejected for D=%d ne01=%d sinks=%d softcap=%.6g fp8=%d; "
                    "using v2/fallback\n",
                    D, ne01, params.sinks != nullptr, params.logit_softcap, (int) params.kv_is_fp8);
        }

        if (use_xmx_v1_path) {
            // v1 kernel — kept for A/B regression and simple D=128 PP FA shapes
            // where v2's small tile is much slower. Do not use it for
            // sink/softcap/FP8/TG cases that motivated the deterministic v2
            // default.
            if (ne01 <= 1) {
                GGML_SYCL_KTRACE("fattn_xmx_v1_f16", " D=%d ncols=1 ne01=%d", D, ne01);
                dispatch_debug_kernel("xmx_v1_f16_ncols1");
                DISPATCH_NCOLS(1, launch_fattn_xmx_f16);
            } else if (ne01 <= 2) {
                GGML_SYCL_KTRACE("fattn_xmx_v1_f16", " D=%d ncols=2 ne01=%d", D, ne01);
                dispatch_debug_kernel("xmx_v1_f16_ncols2");
                DISPATCH_NCOLS(2, launch_fattn_xmx_f16);
            } else if (ne01 <= 4) {
                GGML_SYCL_KTRACE("fattn_xmx_v1_f16", " D=%d ncols=4 ne01=%d", D, ne01);
                dispatch_debug_kernel("xmx_v1_f16_ncols4");
                DISPATCH_NCOLS(4, launch_fattn_xmx_f16);
            } else {
                const size_t local_mem_size = (size_t) dev.get_info<sycl::info::device::local_mem_size>();
                const int    batch_kv       = ggml_sycl_fattn_xmx_v1_select_batch_kv(D, 8, local_mem_size);
                GGML_SYCL_KTRACE("fattn_xmx_v1_f16", " D=%d ncols=8 ne01=%d batch_kv=%d local_mem=%zu", D, ne01,
                                 batch_kv, local_mem_size);
                // The large-KV v1 tile is only validated for complete 8-query
                // blocks. Ragged query tails (for example nb=35 in
                // test-backend-ops) can drift above the FA NMSE gate; keep the
                // regular v1 path for those shapes until the large-KV tail path
                // is fixed.
                if (batch_kv == XMX_BATCH_KV_LARGE && ne01 % 8 == 0) {
                    dispatch_debug_kernel("xmx_v1_f16_ncols8_large_kv");
                    DISPATCH_NCOLS_BATCH_KV(8, XMX_BATCH_KV_LARGE, launch_fattn_xmx_f16);
                } else {
                    dispatch_debug_kernel("xmx_v1_f16_ncols8");
                    DISPATCH_NCOLS(8, launch_fattn_xmx_f16);
                }
            }
        } else {
            // v2 kernel — structurally correct, no SLM aliasing, deterministic.
            // Returns false if the fallback leaf's worst-case SLM exceeds the
            // device's local_mem_size; we flip use_xmx so the TILE block below
            // picks up the dispatch. The block below is gated on `!use_xmx`.
            //
            // Prec-hint dispatch (llama.cpp-0kpp3): PREC_F32 forces a float
            // accumulator (matches test-backend-ops's tight NMSE gate and
            // llama-graph.cpp's unconditional PREC_F32 pin). PREC_DEFAULT picks
            // up the build-time `afloat` typedef — half under GGML_SYCL_F16,
            // float otherwise. Mirrors CUDA's pattern at
            // ggml/src/ggml-cuda/fattn-wmma-f16.cu:541.
            bool       v2_dispatched = false;
            const bool use_f32_acc   = (params.prec == GGML_PREC_F32 || params.max_bias != 0.0f);
            if (ne01 <= 1) {
                GGML_SYCL_KTRACE("fattn_xmx_v2_f16", " D=%d ncols=1 ne01=%d prec=%d", D, ne01, (int) params.prec);
                if (use_f32_acc) {
                    DISPATCH_NCOLS_CTX_ACC(1, launch_fattn_xmx_v2_f16, float);
                } else {
                    DISPATCH_NCOLS_CTX_ACC(1, launch_fattn_xmx_v2_f16, afloat);
                }
                dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_ncols1" : "xmx_v2_f16_ncols1_rejected");
            } else if (ne01 <= 2) {
                GGML_SYCL_KTRACE("fattn_xmx_v2_f16", " D=%d ncols=2 ne01=%d prec=%d", D, ne01, (int) params.prec);
                if (use_f32_acc) {
                    DISPATCH_NCOLS_CTX_ACC(2, launch_fattn_xmx_v2_f16, float);
                } else {
                    DISPATCH_NCOLS_CTX_ACC(2, launch_fattn_xmx_v2_f16, afloat);
                }
                dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_ncols2" : "xmx_v2_f16_ncols2_rejected");
            } else if (ne01 <= 4) {
                GGML_SYCL_KTRACE("fattn_xmx_v2_f16", " D=%d ncols=8 ne01=%d prec=%d", D, ne01, (int) params.prec);
                if (use_f32_acc) {
                    DISPATCH_NCOLS_CTX_ACC(8, launch_fattn_xmx_v2_f16, float);
                } else {
                    DISPATCH_NCOLS_CTX_ACC(8, launch_fattn_xmx_v2_f16, afloat);
                }
                dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_ncols8" : "xmx_v2_f16_ncols8_rejected");
            } else if (ne01 <= 16) {
                GGML_SYCL_KTRACE("fattn_xmx_v2_f16", " D=%d ncols=16 ne01=%d prec=%d", D, ne01, (int) params.prec);
                if (use_f32_acc) {
                    DISPATCH_NCOLS_CTX_ACC(16, launch_fattn_xmx_v2_f16, float);
                } else {
                    DISPATCH_NCOLS_CTX_ACC(16, launch_fattn_xmx_v2_f16, afloat);
                }
                dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_ncols16" : "xmx_v2_f16_ncols16_rejected");
            } else {
                GGML_SYCL_KTRACE("fattn_xmx_v2_f16", " D=%d ncols=16 ne01=%d prec=%d", D, ne01, (int) params.prec);
                if (use_f32_acc) {
                    DISPATCH_NCOLS_CTX_ACC(16, launch_fattn_xmx_v2_f16, float);
                } else {
                    DISPATCH_NCOLS_CTX_ACC(16, launch_fattn_xmx_v2_f16, afloat);
                }
                dispatch_debug_kernel(v2_dispatched ? "xmx_v2_f16_ncols16_large" : "xmx_v2_f16_ncols16_large_rejected");
            }
            if (!v2_dispatched) {
                use_xmx = false;
            }
        }
    }
    if (!use_xmx) {
        // TILE F16 kernel - scalar-SLM-tile fallback for safe_decode / non-XMX GPUs
        //                  / v2 SLM-fit failure (see v2_dispatched flip above).
        if (ne01 <= 1) {
            GGML_SYCL_KTRACE("fattn_tile_f16", " D=%d ncols=1 ne01=%d", D, ne01);
            dispatch_debug_kernel("tile_f16_ncols1");
            DISPATCH_NCOLS(1, launch_fattn_tile_f16);
        } else if (ne01 <= 2) {
            GGML_SYCL_KTRACE("fattn_tile_f16", " D=%d ncols=2 ne01=%d", D, ne01);
            dispatch_debug_kernel("tile_f16_ncols2");
            DISPATCH_NCOLS(2, launch_fattn_tile_f16);
        } else if (ne01 <= 4) {
            GGML_SYCL_KTRACE("fattn_tile_f16", " D=%d ncols=4 ne01=%d", D, ne01);
            dispatch_debug_kernel("tile_f16_ncols4");
            DISPATCH_NCOLS(4, launch_fattn_tile_f16);
        } else {
            GGML_SYCL_KTRACE("fattn_tile_f16", " D=%d ncols=8 ne01=%d", D, ne01);
            dispatch_debug_kernel("tile_f16_ncols8");
            DISPATCH_NCOLS(8, launch_fattn_tile_f16);
        }
    }

#undef DISPATCH_NCOLS
#undef DISPATCH_NCOLS_CTX_ACC
#undef DISPATCH_NCOLS_CTX_ACC_VARIANT
}

// Main flash attention entry point
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor safe_dst) {
    GGML_SYCL_PROFILE_SCOPE_FA("flash_attn");
// Initialize configuration on first call
#if GGML_SYCL_FA_V2_ENABLED
#endif
    init_kv_fp8_config();
    init_fa_esimd_config();
    init_fa_safe_decode_config();
#if GGML_SYCL_DNNL
    init_fa_onednn_config();
#endif

    // One-shot build-flag announcement — independent of oneDNN so builds with
    // GGML_SYCL_F16=ON + GGML_SYCL_DNN=OFF still see it.
#ifdef GGML_SYCL_F16
    {
        static bool s_f16_banner_logged = false;
        if (!s_f16_banner_logged) {
            s_f16_banner_logged = true;
            fprintf(stderr, "[SYCL] GGML_SYCL_F16 build: attention Q/accumulators are f16\n");
        }
    }
#endif

    const ggml_tensor * dst = safe_dst.raw();
    auto                Q_t = safe_dst.src(0);
    auto                K_t = safe_dst.src(1);
    auto                V_t = safe_dst.src(2);
    const ggml_tensor * Q   = Q_t.raw();
    const ggml_tensor * K   = K_t.raw();
    const ggml_tensor * V   = V_t.raw();
    // Flash attention uses activation tensors (Q/K/V + KV cache). No weight tensors
    // are expected here, so unified-cache streaming is intentionally skipped.
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(Q));
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(K));
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(V));

    // Check tiered dispatch for K/V cache tensors
    if (g_tiered_enabled.load(std::memory_order_relaxed)) {
        // Check K tensor (name[0] checks if name is non-empty)
        if (K && K->name[0]) {
            ggml_sycl::memory_tier tier;
            bool                   found = false;
            void *                 ptr   = get_cached_tensor_ptr(K->name, &tier, &found);
            if (ptr) {
                GGML_SYCL_DEBUG("[SYCL] fattn K tiered hit: %s (tier=%d)\n", K->name, static_cast<int>(tier));
            } else if (found) {
                GGML_SYCL_DEBUG("[SYCL] fattn K tiered pending: %s\n", K->name);
            }
        }

        // Check V tensor (name[0] checks if name is non-empty)
        if (V && V->name[0]) {
            ggml_sycl::memory_tier tier;
            bool                   found = false;
            void *                 ptr   = get_cached_tensor_ptr(V->name, &tier, &found);
            if (ptr) {
                GGML_SYCL_DEBUG("[SYCL] fattn V tiered hit: %s (tier=%d)\n", V->name, static_cast<int>(tier));
            } else if (found) {
                GGML_SYCL_DEBUG("[SYCL] fattn V tiered pending: %s\n", V->name);
            }
        }
    }

    // Event-based synchronization moved to ubatch level (llama-context.cpp)
    // This flash_attn no longer waits on individual events - the wait happens
    // at the START of each ubatch before any kernels are submitted.
    // This ensures all previous ubatch's KV writes complete before this ubatch starts.
    const ggml_tensor * mask       = dst->src[3];
    const ggml_tensor * sinks      = dst->src[4];  // Attention sinks tensor (may be null)
    const ggml_tensor * q_seq_ids  = dst->src[5];  // Sequence IDs for query tokens (may be null)
    const ggml_tensor * kv_seq_ids = dst->src[6];  // Sequence IDs for KV positions (may be null)

    // l144i probe: stage all inputs to fattn and log their hashes to isolate race.
    // Proved (2026-04-20): Q/K/V/mask/sinks are bit-identical across runs; fattn
    // OUTPUT differs.  Non-determinism is INSIDE the XMX kernel (not input staging).
    if (::ggml_sycl::l144i::enabled() && ctx.stream() && dst && dst->name[0]) {
        auto probe = [&](const char * site, const ggml_tensor * t) {
            if (!t) {
                return;
            }
            void * ptr = ggml_sycl_resolve_tensor_ptr(const_cast<ggml_tensor *>(t), ctx.device);
            if (!ptr) {
                return;
            }
            const std::size_t bytes = static_cast<std::size_t>(ggml_nbytes(t));
            if (bytes == 0 || bytes > 16u * 1024u * 1024u) {
                return;
            }
            std::vector<char> buf(bytes);
            ctx.stream()->memcpy(buf.data(), ptr, bytes).wait();
            if (t->type == GGML_TYPE_F32) {
                GGML_SYCL_L144I_PROBE_FLOATS(site, t->name[0] ? t->name : "?", -1, -1,
                                             reinterpret_cast<const float *>(buf.data()), bytes / sizeof(float));
            } else {
                GGML_SYCL_L144I_PROBE_BYTES(site, t->name[0] ? t->name : "?", -1, -1, buf.data(), bytes);
            }
        };
        probe("fa/src0_Q", Q);
        probe("fa/src1_K", K);
        probe("fa/src2_V", V);
        probe("fa/src3_mask", mask);
        probe("fa/src4_sinks", sinks);
    }

    GGML_ASSERT(Q->type == GGML_TYPE_F32 || Q->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == GGML_TYPE_F16 || ggml_sycl_type_is_fp8_e4m3(K->type));  // FP16 or FP8 KV cache
    GGML_ASSERT(V->type == GGML_TYPE_F16 || ggml_sycl_type_is_fp8_e4m3(V->type));  // FP16 or FP8 KV cache
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Extract scale, max_bias, and logit_softcap from op_params
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Read use_paged_layout from op_params[4] (set by ggml_flash_attn_ext_set_paged_layout)
    // op_params layout: [0-2]=float scale/max_bias/logit_softcap, [3]=prec, [4]=use_paged_layout
    const int32_t use_paged_layout_i32 = ((const int32_t *) dst->op_params)[4];
    const bool    use_paged_layout     = (use_paged_layout_i32 != 0);

    // If using logit_softcap, adjust scale
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Calculate ALiBi parameters
    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Build the params structure
    fattn_params params;

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int device         = ctx.device;
    params.Q                 = static_cast<const char *>(Q_t.resolve_ptr());
    params.K                 = static_cast<const char *>(K_t.resolve_ptr());
    {
        size_t              k_view_offs = 0;
        const ggml_tensor * k_root      = ggml_sycl_fattn_xmx_root_tensor(K, &k_view_offs);
        auto                k_handle    = ggml_sycl_fattn_xmx_root_data_handle(k_root, device);
        params.K_handle_valid           = k_handle.valid();
        params.K_handle_hash            = params.K_handle_valid ? k_handle.hash() : 0;
        params.K_view_offs              = k_view_offs;
    }
    params.V                 = static_cast<const char *>(V_t.resolve_ptr());
    // llama.cpp-15li2 CRIT-1: FA's src[4] (`sinks`) is a per-layer WEIGHT
    // for GPT-OSS; with host-planned layers it resolves to host-pinned
    // USM.  Stage to device if the resolved pointer is not
    // sycl::usm::alloc::device.  `mask` is defensively staged as well
    // even though today it is always a per-batch activation — if a
    // future model makes its mask weight-planned, this will keep working.
    sycl::queue * stream_ptr = ctx.stream();
    params.mask =
        mask ? ggml_sycl_fattn_stage_weight_src(mask, device, stream_ptr, "mask", g_tl_fattn_weight_stage.mask_dev,
                                                g_tl_fattn_weight_stage.mask_capacity) :
               nullptr;
    params.sinks = sinks ? ggml_sycl_fattn_stage_weight_src(sinks, device, stream_ptr, "attn_sinks",
                                                            g_tl_fattn_weight_stage.sinks_dev,
                                                            g_tl_fattn_weight_stage.sinks_capacity) :
                           nullptr;
    params.dst   = safe_dst.resolve_as<float>();

    // Source tensor types — consumers (e.g. oneDNN SDPA) need these to size byte
    // strides correctly and to build logical tensors with the right data_type.
    params.Q_type    = Q->type;
    params.K_type    = K->type;
    params.V_type    = V->type;
    params.mask_type = mask ? mask->type : GGML_TYPE_F32;

    params.scale         = scale;
    params.max_bias      = max_bias;
    params.m0            = m0;
    params.m1            = m1;
    params.n_head_log2   = n_head_log2;
    params.logit_softcap = logit_softcap;

    // Per-op precision hint. test-backend-ops sets PREC_F32 on every case via
    // ggml_flash_attn_ext_set_prec; llama-graph.cpp:1726 pins PREC_F32
    // unconditionally for the SYCL/CUDA paths. The XMX-v2 dispatcher reads
    // this to force a float-accumulator specialization on PREC_F32.
    params.prec = ggml_flash_attn_ext_get_prec(dst);

    // Q dimensions: [batch, n_heads, n_queries, head_dim]
    params.ne00 = Q->ne[0];  // head_dim
    params.ne01 = Q->ne[1];  // n_queries
    params.ne02 = Q->ne[2];  // n_heads
    params.ne03 = Q->ne[3];  // batch

    params.nb01 = Q->nb[1];
    params.nb02 = Q->nb[2];
    params.nb03 = Q->nb[3];

    // K dimensions: [batch, n_kv_heads, n_kv, head_dim]
    params.ne10 = K->ne[0];  // head_dim
    params.ne11 = K->ne[1];  // n_kv (sequence length)
    params.ne12 = K->ne[2];  // n_kv_heads
    params.ne13 = K->ne[3];  // batch

    params.nb11 = K->nb[1];
    params.nb12 = K->nb[2];
    params.nb13 = K->nb[3];

    // V strides
    params.nb21 = V->nb[1];
    params.nb22 = V->nb[2];
    params.nb23 = V->nb[3];

    // Mask dimensions and strides (if present)
    // mask layout: [ne3, ne2, ne1, ne0] = [batch, heads, n_tokens_padded, n_kv]
    if (mask) {
        params.ne30 = mask->ne[0];  // n_kv
        params.ne31 = mask->ne[1];  // n_tokens_padded
        params.ne32 = mask->ne[2];  // heads
        params.ne33 = mask->ne[3];  // batch
        params.nb31 = mask->nb[1];
        params.nb32 = mask->nb[2];
        params.nb33 = mask->nb[3];

        // Assertion: ne11 (K seq len) should equal ne30 (mask's first dim)
        GGML_ASSERT(params.ne11 == params.ne30 && "K sequence length must match mask dimension");
    } else {
        params.ne30 = 0;
        params.ne31 = 0;
        params.ne32 = 0;
        params.ne33 = 0;
        params.nb31 = 0;
        params.nb32 = 0;
        params.nb33 = 0;
    }

    // Multi-sequence batching support
    // Use sequence ID tensors from src[5] and src[6] to enable cross-sequence skipping
    // This optimization reduces wasted computation when queries only need their own sequence's KV
    dpct::queue_ptr stream = ctx.stream();

    // Initialize legacy offset-based multi-seq to disabled
    params.n_seqs         = 0;
    params.seq_q_offsets  = nullptr;
    params.seq_kv_offsets = nullptr;

    // Thread-local storage for sequence boundary offsets
    // These persist across calls within the same thread
    thread_local std::vector<int32_t> tl_seq_q_offsets;
    thread_local std::vector<int32_t> tl_seq_kv_offsets;

    // Thread-local device buffers for seq_ids (reused across calls)
    // We need to manage these manually since host tensor->data can't be accessed from GPU
    tl_seq_id_buffers & tl_buffers = g_tl_seq_buffers;

    // Use sequence ID tensors if provided
    // Skip during SYCL graph recording - this optimization requires wait() calls
    // which are not allowed during graph recording
    if (!g_ggml_sycl_graph_recording && q_seq_ids && kv_seq_ids && q_seq_ids->type == GGML_TYPE_I32 &&
        kv_seq_ids->type == GGML_TYPE_I32) {
        const size_t q_size  = q_seq_ids->ne[0] * sizeof(int32_t);
        const size_t kv_size = kv_seq_ids->ne[0] * sizeof(int32_t);

        // Check if tensors are on host buffers (need to copy to device)
        bool q_on_host  = q_seq_ids->buffer ? ggml_backend_buffer_is_host(q_seq_ids->buffer) : true;
        bool kv_on_host = kv_seq_ids->buffer ? ggml_backend_buffer_is_host(kv_seq_ids->buffer) : true;

        auto resolve_host_seq_ids = [&](const ggml_tensor * tensor) -> const int32_t * {
            void * ptr = ggml_sycl_resolve_tensor_ptr(tensor, device);
            if (ptr) {
                try {
                    if (ggml_sycl_get_alloc_type(ptr) != sycl::usm::alloc::device) {
                        return static_cast<const int32_t *>(ptr);
                    }
                } catch (...) {
                    GGML_LOG_WARN("[SYCL] resolve_host_seq_ids: alloc type query failed, falling back to host data\n");
                }
            }
            return static_cast<const int32_t *>(ggml_sycl_host_data(tensor));
        };

        // Get the host pointers from thread-local cache (set by llama-kv-cache.cpp)
        // These are USM host pointers that are accessible from both CPU and GPU
        size_t          cached_q_count = 0, cached_kv_count = 0;
        const int32_t * cached_q_seq_ids  = ggml_sycl_get_seq_ids_host_q(&cached_q_count);
        const int32_t * cached_kv_seq_ids = ggml_sycl_get_seq_ids_host_kv(&cached_kv_count);

        // If tensors are on device (not host), use the cached host pointers instead
        // The scheduler creates device tensors but doesn't copy INPUT data, so tensor->data is invalid
        // However, the llama layer has set the USM host pointers via ggml_backend_sycl_set_seq_ids_host()
        const int32_t * host_q_ptr  = q_on_host ? resolve_host_seq_ids(q_seq_ids) : cached_q_seq_ids;
        const int32_t * host_kv_ptr = kv_on_host ? resolve_host_seq_ids(kv_seq_ids) : cached_kv_seq_ids;

        if (!host_q_ptr || !host_kv_ptr) {
            // No valid host pointers available - skip optimization
            static bool warned_no_cache = false;
            if (!warned_no_cache) {
                fprintf(stderr, "[SEQ_IDS] WARNING: Device tensors detected but no cached host pointers\n");
                fprintf(stderr, "[SEQ_IDS]   Falling back to mask-based sequence detection\n");
                warned_no_cache = true;
            }
            // params.q_seq_ids and params.kv_seq_ids remain nullptr (default)
        } else {
            // We have valid host pointers (either from tensor or from cache)
            // Need to copy from host to device for kernel access
            // Reallocate device buffers if size changed or first time
            if (tl_buffers.alloc_queue != stream || tl_buffers.q_seq_ids_size < q_size) {
                tl_buffers.free_ptr(tl_buffers.q_seq_ids_dev, tl_buffers.q_seq_ids_size);
                tl_buffers.q_seq_ids_dev  = tl_buffers.alloc_ptr(q_seq_ids->ne[0], stream);
                tl_buffers.q_seq_ids_size = q_size;
            }
            if (tl_buffers.alloc_queue != stream || tl_buffers.kv_seq_ids_size < kv_size) {
                tl_buffers.free_ptr(tl_buffers.kv_seq_ids_dev, tl_buffers.kv_seq_ids_size);
                tl_buffers.kv_seq_ids_dev  = tl_buffers.alloc_ptr(kv_seq_ids->ne[0], stream);
                tl_buffers.kv_seq_ids_size = kv_size;
            }
            tl_buffers.alloc_queue = stream;

            // Copy from host (USM) to device using the correct host pointers
            // Note: In-order queue ensures memcpy completes before subsequent kernel launch
            ggml_sycl_trace_memcpy_during_recording("fattn.cpp:q_seq_ids", q_size);
            ggml_sycl_graph_safe_memcpy(*stream, tl_buffers.q_seq_ids_dev, host_q_ptr, q_size);
            ggml_sycl_trace_memcpy_during_recording("fattn.cpp:kv_seq_ids", kv_size);
            ggml_sycl_graph_safe_memcpy(*stream, tl_buffers.kv_seq_ids_dev, host_kv_ptr, kv_size);

            params.q_seq_ids  = tl_buffers.q_seq_ids_dev;
            params.kv_seq_ids = tl_buffers.kv_seq_ids_dev;

            // Compute sequence boundaries from the seq_ids arrays
            // This enables the kernel to skip cross-sequence KV computation entirely
            // Note: We use the HOST pointers for boundary computation (which happens on CPU)
            int n_queries = static_cast<int>(q_seq_ids->ne[0]);
            int n_kv      = static_cast<int>(kv_seq_ids->ne[0]);

            int n_seqs = compute_sequence_boundaries_from_ids(host_q_ptr, n_queries, host_kv_ptr, n_kv,
                                                              tl_seq_q_offsets, tl_seq_kv_offsets);

            // Copy sequence boundary offsets to device memory for kernel access
            // This enables the kernel to skip entire KV blocks for non-matching sequences
            if (n_seqs > 1) {
                const size_t offsets_size = (n_seqs + 1) * sizeof(int32_t);

                // Reallocate device buffers if capacity is insufficient
                if (tl_buffers.seq_offsets_capacity < offsets_size) {
                    tl_buffers.free_ptr(tl_buffers.seq_q_offsets_dev, tl_buffers.seq_offsets_capacity);
                    tl_buffers.free_ptr(tl_buffers.seq_kv_offsets_dev, tl_buffers.seq_offsets_capacity);
                    tl_buffers.seq_q_offsets_dev    = tl_buffers.alloc_ptr(n_seqs + 1, stream);
                    tl_buffers.seq_kv_offsets_dev   = tl_buffers.alloc_ptr(n_seqs + 1, stream);
                    tl_buffers.seq_offsets_capacity = offsets_size;
                }

                // Copy offsets to device
                // Note: In-order queue ensures memcpy completes before subsequent kernel launch
                ggml_sycl_trace_memcpy_during_recording("fattn.cpp:seq_q_offsets", offsets_size);
                ggml_sycl_graph_safe_memcpy(*stream, tl_buffers.seq_q_offsets_dev, tl_seq_q_offsets.data(),
                                            offsets_size);
                ggml_sycl_trace_memcpy_during_recording("fattn.cpp:seq_kv_offsets", offsets_size);
                ggml_sycl_graph_safe_memcpy(*stream, tl_buffers.seq_kv_offsets_dev, tl_seq_kv_offsets.data(),
                                            offsets_size);

                // Set params for kernel to use sequence boundary optimization
                params.n_seqs         = n_seqs;
                params.seq_q_offsets  = tl_buffers.seq_q_offsets_dev;
                params.seq_kv_offsets = tl_buffers.seq_kv_offsets_dev;
            }
        }
    } else {
        params.q_seq_ids  = nullptr;
        params.kv_seq_ids = nullptr;
    }

    // PagedAttention support
    // Block table (src[7]) and seq_lens (src[8]) are set via ggml_flash_attn_ext_set_paged()
    const ggml_tensor * block_table     = dst->src[7];
    const ggml_tensor * seq_lens_tensor = dst->src[8];

    if (block_table && seq_lens_tensor) {
        // Enable PagedAttention: K/V are stored in blocks, accessed via block_table
        params.use_paged_attn     = true;
        params.block_size         = 16;  // Fixed block size (matches vLLM and XMX tile size)
        // block_table shape: [max_blocks, n_seqs] where ne[0]=max_blocks (columns), ne[1]=n_seqs (rows)
        // Kernel access: block_table[seq * max_blocks + block] requires max_blocks = ne[0]
        params.max_blocks_per_seq = static_cast<int32_t>(block_table->ne[0]);
        params.block_table        = (const int32_t *) ggml_sycl_get_data_ptr(block_table, device);
        params.seq_lens           = (const int32_t *) ggml_sycl_get_data_ptr(seq_lens_tensor, device);

#if 0  // Debug output disabled \
       // Print only once to avoid flooding output
        static bool paged_attn_info_shown = false;
        if (!paged_attn_info_shown) {
            fprintf(stderr, "[SYCL] PagedAttention enabled: block_size=%d, max_blocks=%d\n",
                    params.block_size, params.max_blocks_per_seq);
            paged_attn_info_shown = true;
        }
#endif
    } else {
        // Standard contiguous K/V mode
        params.use_paged_attn     = false;
        params.block_size         = 16;
        params.max_blocks_per_seq = 0;
        params.block_table        = nullptr;
        params.seq_lens           = nullptr;
    }

    // Set paged layout flag (read from op_params[4], set via ggml_flash_attn_ext_set_paged_layout)
    params.use_paged_layout = use_paged_layout;

    if (g_ggml_sycl_graph_recording && ctx.fa_graph_ptrs_recording) {
        ggml_backend_sycl_context::fa_graph_ptr_snapshot snap;
        snap.q           = params.Q;
        snap.k           = params.K;
        snap.v           = params.V;
        snap.dst         = params.dst;
        snap.mask        = params.mask;
        snap.sinks       = params.sinks;
        snap.block_table = params.block_table;
        snap.seq_lens    = params.seq_lens;
        ctx.fa_graph_ptrs.push_back(snap);
    }

    // Set FP8 KV cache flag - enables on-the-fly dequantization in flash attention kernel
    params.kv_is_fp8 = (ggml_sycl_type_is_fp8_e4m3(K->type) && ggml_sycl_type_is_fp8_e4m3(V->type));

    // Multi-token decode support - disabled by default
    // Will be enabled when q_positions array is provided via thread-local storage
    params.multi_token_decode = false;
    params.q_positions        = nullptr;
    params.kv_base_pos        = 0;

    const int D = Q->ne[0];

#if GGML_SYCL_FA_V2_ENABLED
    // ==========================================================================
    // Paged Attention V2 Dispatch (for long sequences with paged KV layout)
    // ==========================================================================
    // V2 uses multi-partition algorithm when:
    // 1. Paged attention is enabled with block tables
    // 2. GGML_SYCL_PAGED_V2=1 environment variable is set
    // 3. Max sequence length exceeds V2_PARTITION_SIZE (512)
    //
    // NOTE: V2 requires K/V to be stored in paged format:
    //   K/V: [num_blocks, num_kv_heads, block_size, head_dim]
    // This is not yet supported in the current llama.cpp KV cache layout.
    // V2 dispatch is prepared for future paged KV cache implementation.

    const int max_context_len = params.ne11;  // n_kv = sequence length
    bool      use_v2_dispatch = false;
    bool      use_auto_v2     = false;        // Auto-V2: identity block table for contiguous KV

    // Check if V2 dispatch should be used:
    // V2 has two modes:
    // 1. Paged V2: Uses actual block tables from paged KV cache
    // 2. Auto-V2: Generates identity block table for contiguous 3D KV cache
    //
    // Auto-V2 allows using V2's partitioned algorithm benefits (O(n) memory for softmax)
    // without requiring paged KV cache infrastructure changes.
    //
    // NOTE: V2 dispatch is NOW compatible with SYCL command graphs because:
    // 1. Temp buffers are persistent (g_v2_auto.temp_buf) - no malloc/free during dispatch
    // 2. No wait() calls after kernel launch
    // 3. All allocations happen on first use/resize only
    if (g_sycl_paged_v2_enabled && should_use_paged_attention_v2(max_context_len)) {
        if (params.use_paged_attn && params.block_table) {
            // Paged V2: Use actual block tables
            use_v2_dispatch = true;
            use_auto_v2     = false;
        } else {
            // Auto-V2: Generate identity block table for contiguous KV
            use_v2_dispatch                = true;
            use_auto_v2                    = true;
            static bool auto_v2_info_shown = false;
            if (!auto_v2_info_shown) {
                fprintf(stderr, "[SYCL] V2 Auto-Mode: Using identity block table for long sequences\n");
                auto_v2_info_shown = true;
            }
        }
    }

    // V2 dispatch: partition-based attention for long sequences
    // NOTE: V2 is designed for vLLM-style continuous batching where each "seq" in the batch
    // is a separate request with its own KV cache. In llama.cpp's single-sequence mode,
    // all query tokens share the same sequence's KV cache, so we need to check compatibility.

    // Get actual number of sequences from seq_lens tensor (already declared above)
    const int actual_n_seqs    = seq_lens_tensor ? (int) seq_lens_tensor->ne[0] : 1;
    const int num_query_tokens = params.ne01;  // Number of query tokens in batch

    // V2 only works correctly when:
    // 1. We have continuous batching with multiple sequences (actual_n_seqs == num_query_tokens), OR
    // 2. Single sequence mode with one query token per call (num_query_tokens == 1)
    // For prefill (single sequence with many query tokens), fall back to standard kernel
    const bool v2_compatible = (actual_n_seqs == num_query_tokens) || (num_query_tokens == 1);

    if (use_v2_dispatch && !v2_compatible) {
        // Prefill mode: single sequence with multiple query tokens - V2 not designed for this
        // Fall back to standard kernel
        static bool v2_fallback_warned = false;
        if (!v2_fallback_warned) {
            fprintf(stderr, "[SYCL] V2 skipped: prefill mode (n_seqs=%d, n_query=%d) not supported\n", actual_n_seqs,
                    num_query_tokens);
            fprintf(stderr, "[SYCL]   V2 requires continuous batching (n_seqs==n_query) or single-token decode\n");
            v2_fallback_warned = true;
        }
        use_v2_dispatch = false;
    }

    if (use_v2_dispatch) {
        const int num_seqs     = actual_n_seqs;  // Use actual sequence count, not query token count
        const int num_heads    = Q->ne[2];       // Number of query heads
        const int num_kv_heads = K->ne[2];       // Number of KV heads

        // For auto-V2: block_size = 1 (each "block" is one token)
        // For paged V2: use actual block_size from params
        const int block_size         = use_auto_v2 ? 1 : params.block_size;
        const int max_blocks_per_seq = use_auto_v2 ? max_context_len : params.max_blocks_per_seq;

        // Get pointers to block_table and seq_lens
        const int32_t * v2_block_table = nullptr;
        const int32_t * v2_seq_lens    = nullptr;

        if (use_auto_v2) {
            // Auto-V2: Generate identity block table and seq_lens on device
            // block_table[seq][block] = block (identity mapping)
            // seq_lens[seq] = max_context_len (all seqs have same context)
            const size_t block_table_size = num_seqs * max_blocks_per_seq;
            const size_t seq_lens_size    = num_seqs;

            // Reallocate device buffers if needed
            // NOTE: Skip reallocation during SYCL graph recording (malloc/free forbidden)
            // This assumes buffers were allocated in prior decode steps before graph recording
            bool needs_realloc_block =
                !g_ggml_sycl_graph_recording &&
                (g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.block_table_capacity < block_table_size);
            bool needs_realloc_seq = !g_ggml_sycl_graph_recording && (g_v2_auto.alloc_queue != ctx.stream() ||
                                                                      g_v2_auto.seq_lens_capacity < seq_lens_size);

            if (needs_realloc_block) {
                if (g_v2_auto.block_table && g_v2_auto.alloc_queue) {
                    sycl::free(g_v2_auto.block_table, *g_v2_auto.alloc_queue);
                }
                g_v2_auto.block_table =
                    ggml_sycl_malloc_device_t<int32_t>(block_table_size, *ctx.stream(), "fattn_block_table");
                g_v2_auto.block_table_capacity = block_table_size;
            }
            if (needs_realloc_seq) {
                if (g_v2_auto.seq_lens && g_v2_auto.alloc_queue) {
                    sycl::free(g_v2_auto.seq_lens, *g_v2_auto.alloc_queue);
                }
                g_v2_auto.seq_lens = ggml_sycl_malloc_device_t<int32_t>(seq_lens_size, *ctx.stream(), "fattn_seq_lens");
                g_v2_auto.seq_lens_capacity = seq_lens_size;
            }
            if (needs_realloc_block || needs_realloc_seq) {
                g_v2_auto.alloc_queue = ctx.stream();
            }

            // Fill identity block table: block_table[i] = i
            // Use a kernel for efficiency (parallel fill)
            ctx.stream()->parallel_for<class fattn_v2_fill_block_table_runtime_kernel>(
                sycl::range<1>(block_table_size), [block_table_ptr = g_v2_auto.block_table](sycl::id<1> idx) {
                    block_table_ptr[idx] = static_cast<int32_t>(idx[0]);
                });

            // Fill seq_lens: all sequences have the same context length
            ctx.stream()->parallel_for<class fattn_v2_fill_seq_lens_kernel>(
                sycl::range<1>(seq_lens_size), [seq_lens_ptr = g_v2_auto.seq_lens, ctx_len = max_context_len](
                                                   sycl::id<1> idx) { seq_lens_ptr[idx] = ctx_len; });

            // Note: In-order queue ensures parallel_for fills complete before V2 kernel

            v2_block_table = g_v2_auto.block_table;
            v2_seq_lens    = g_v2_auto.seq_lens;
        } else {
            // Paged V2: Use provided block_table and seq_lens
            v2_block_table = params.block_table;
            v2_seq_lens    = params.seq_lens;
        }

        // Get temporary buffer sizes
        const size_t temp_size = paged_attention_v2_temp_size(num_seqs, num_heads, max_context_len, D);

        // Use persistent temp buffer (avoids malloc/free during SYCL graph recording)
        // Reallocate if needed (buffer grows but never shrinks)
        // NOTE: Skip reallocation during SYCL graph recording (malloc/free forbidden)
        bool needs_realloc_temp = !g_ggml_sycl_graph_recording &&
                                  (g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.temp_buf_capacity < temp_size);
        if (needs_realloc_temp) {
            if (g_v2_auto.temp_buf && g_v2_auto.alloc_queue) {
                sycl::free(g_v2_auto.temp_buf, *g_v2_auto.alloc_queue);
            }
            g_v2_auto.temp_buf          = ggml_sycl_malloc_device_t<uint8_t>(temp_size, *ctx.stream(), "fattn_temp");
            g_v2_auto.temp_buf_capacity = temp_size;
            g_v2_auto.alloc_queue       = ctx.stream();
        }

        // If we're graph recording and buffers don't exist, we can't proceed
        // This shouldn't happen if the first decode step ran before graph recording
        if (!g_v2_auto.temp_buf) {
            // Fall back to standard dispatch path
            static bool v2_graph_warning = false;
            if (!v2_graph_warning) {
                fprintf(stderr,
                        "[SYCL] WARNING: V2 temp buffer not allocated during graph recording, "
                        "falling back to standard FA\n");
                v2_graph_warning = true;
            }
            use_v2_dispatch = false;
        }

        // If V2 dispatch was disabled (e.g., no temp buffer during graph recording),
        // we'll fall through to standard dispatch after the #endif

        if (use_v2_dispatch) {
            // Use device pointers from params (obtained via ggml_sycl_get_data_ptr)
            float *            out   = params.dst;
            const sycl::half * Q_dev = (const sycl::half *) params.Q;
            const char *       K_dev = params.K;
            const char *       V_dev = params.V;

#    if SYCL_V2_ESIMD_AVAILABLE
            // ESIMD V2 kernel: optimized single-kernel version with SIMD vectorization
            // Preferred over scalar V2 when ESIMD is available (25-45% faster)
            // Uses element-index addressing (same as working ESIMD kernel)
            if (g_sycl_fa_esimd_enabled && v2_esimd_available()) {
                GGML_SYCL_DEBUG(
                    "[SYCL] V2 ESIMD dispatch: num_seqs=%d num_heads=%d num_kv_heads=%d D=%d "
                    "max_context=%d block_size=%d auto_v2=%d\n",
                    num_seqs, num_heads, num_kv_heads, D, max_context_len, block_size, use_auto_v2 ? 1 : 0);

                static bool v2_esimd_info_shown = false;
                if (!v2_esimd_info_shown) {
                    fprintf(stderr, "[SYCL] Using ESIMD-optimized V2 attention kernel\n");
                    v2_esimd_info_shown = true;
                }

                // Dispatch ESIMD V2 kernel based on head dimension and Q type
                // Q can be F32 (decode mode) or F16 (some models) - need correct template parameter
                // New signature uses params struct for proper element-index addressing
                if (Q->type == GGML_TYPE_F32) {
                    switch (D) {
                        case 64:
                            launch_v2_esimd_attention<64, float>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                 max_blocks_per_seq, max_context_len, block_size,
                                                                 ctx.stream());
                            break;
                        case 128:
                            launch_v2_esimd_attention<128, float>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                  max_blocks_per_seq, max_context_len, block_size,
                                                                  ctx.stream());
                            break;
                        default:
                            // D=256 not yet implemented in ESIMD V2, fall through to scalar V2
                            goto scalar_v2_dispatch;
                    }
                } else {
                    // Q is F16
                    switch (D) {
                        case 64:
                            launch_v2_esimd_attention<64, sycl::half>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                      max_blocks_per_seq, max_context_len, block_size,
                                                                      ctx.stream());
                            break;
                        case 128:
                            launch_v2_esimd_attention<128, sycl::half>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                       max_blocks_per_seq, max_context_len, block_size,
                                                                       ctx.stream());
                            break;
                        default:
                            // D=256 not yet implemented in ESIMD V2, fall through to scalar V2
                            goto scalar_v2_dispatch;
                    }
                }

                return;  // ESIMD V2 dispatch complete
            }
scalar_v2_dispatch:
#    endif  // SYCL_V2_ESIMD_AVAILABLE

            // Scalar V2 kernel: fallback when ESIMD is not available
            void * temp_buf = g_v2_auto.temp_buf;

            const int    max_num_partitions = (max_context_len + V2_PARTITION_SIZE - 1) / V2_PARTITION_SIZE;
            const size_t exp_sums_size      = num_seqs * num_heads * max_num_partitions * sizeof(float);

            float * exp_sums   = (float *) temp_buf;
            float * max_logits = exp_sums + num_seqs * num_heads * max_num_partitions;
            float * tmp_out    = max_logits + num_seqs * num_heads * max_num_partitions;

            GGML_SYCL_DEBUG(
                "[SYCL] V2 scalar dispatch: num_seqs=%d num_heads=%d num_kv_heads=%d D=%d "
                "max_context=%d partitions=%d block_size=%d auto_v2=%d\n",
                num_seqs, num_heads, num_kv_heads, D, max_context_len, max_num_partitions, block_size,
                use_auto_v2 ? 1 : 0);

            // Dispatch scalar V2 kernel based on head dimension
            switch (D) {
                case 64:
                    launch_paged_attention_v2<64>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev, params.scale,
                                                  v2_block_table, v2_seq_lens, num_seqs, num_heads, num_kv_heads,
                                                  max_blocks_per_seq, max_context_len, block_size, params.nb11,
                                                  params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                case 128:
                    launch_paged_attention_v2<128>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev,
                                                   params.scale, v2_block_table, v2_seq_lens, num_seqs, num_heads,
                                                   num_kv_heads, max_blocks_per_seq, max_context_len, block_size,
                                                   params.nb11, params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                case 256:
                    launch_paged_attention_v2<256>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev,
                                                   params.scale, v2_block_table, v2_seq_lens, num_seqs, num_heads,
                                                   num_kv_heads, max_blocks_per_seq, max_context_len, block_size,
                                                   params.nb11, params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                default:
                    GGML_ABORT("Unsupported head dimension for V2 attention: %d", D);
            }

            // Note: No wait() or free() here - temp buffer is persistent (g_v2_auto.temp_buf)
            // This allows V2 dispatch to work with SYCL command graphs

            return;  // V2 dispatch complete
        }  // if (use_v2_dispatch) - inner check after potential fallback
    }  // if (use_v2_dispatch) - main V2 execution block (line 893)
#endif  // GGML_SYCL_FA_V2_ENABLED

    // ==========================================================================
    // Standard Dispatch (XMX or MMA-F16 kernels)
    // ==========================================================================

    if (Q->type == GGML_TYPE_F32) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, float>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, float>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, float>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else if (Q->type == GGML_TYPE_F16) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, sycl::half>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, sycl::half>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, sycl::half>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else {
        GGML_ABORT("Unsupported Q type for SYCL flash attention");
    }

    // Buffer aliasing debug: trace flash attention output pointer and values
    // This helps identify if FA writes to expected buffer vs what MUL_MAT reads
    // NOTE: Only enabled when graphs are disabled (wait() is incompatible with graph recording)
    {
        static bool fa_buf_debug    = getenv("GGML_SYCL_BUFFER_ALIAS_DEBUG") != nullptr;
        static bool graphs_disabled = getenv("GGML_SYCL_DISABLE_GRAPH") != nullptr;
        if (fa_buf_debug && graphs_disabled) {
            static int fa_trace_count = 0;
            fa_trace_count++;
            if (fa_trace_count <= 10) {
                auto * stream = ctx.stream();
                stream->wait();  // Ensure kernel completes

                // Read first 4 floats from dst
                float  host_vals[4]   = { 0 };
                size_t total_elements = params.ne01 * params.ne02 * D;  // n_queries * n_heads * head_dim
                if (total_elements >= 4) {
                    stream->memcpy(host_vals, params.dst, 4 * sizeof(float)).wait();
                }

                bool is_zeros =
                    (host_vals[0] == 0.0f && host_vals[1] == 0.0f && host_vals[2] == 0.0f && host_vals[3] == 0.0f);

                fprintf(stderr,
                        "[FA_OUTPUT] %s dst=%p first4=[%.4f, %.4f, %.4f, %.4f] is_zeros=%d "
                        "ne01=%lld ne02=%lld D=%d\n",
                        dst->name, (void *) params.dst, host_vals[0], host_vals[1], host_vals[2], host_vals[3],
                        is_zeros ? 1 : 0, (long long) params.ne01, (long long) params.ne02, D);
            }
        }
    }

    // Debug dumping controlled by GGML_SYCL_FA_DEBUG environment variable
    // Level 1: Basic inputs/outputs for first few heads
    // Level 2: Verbose mode with all heads and intermediate values
    if (fattn_debug_level() > 0) {
        // Wait for kernel to complete
        stream->wait();

        // Track call count
        static int fa_call_count = 0;
        fa_call_count++;

        // Only dump first N calls
        const int max_dumps = 20;
        if (fa_call_count <= max_dumps) {
            auto & dbg     = get_fattn_debug_ctx();
            dbg.call_id    = fa_call_count;
            dbg.n_queries  = params.ne01;
            dbg.n_heads    = params.ne02;
            dbg.n_kv_heads = params.ne12;
            dbg.n_kv       = params.ne11;
            dbg.D          = D;
            dbg.scale      = params.scale;
            dbg.is_fa_on   = true;

            dbg.open_file("on");

            int Q_type_size = (Q->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(sycl::half);
            fattn_debug_dump_Q(stream, params.Q, Q_type_size, D, params.ne01, params.ne02, params.nb01, params.nb02,
                               params.scale);
            fattn_debug_dump_K(stream, params.K, D, params.ne11, params.ne12, params.nb11, params.nb12);
            fattn_debug_dump_V(stream, params.V, D, params.ne11, params.ne12, params.nb21, params.nb22);
            fattn_debug_dump_mask(stream, params.mask, params.ne30, params.ne01, params.nb31, params.ne30);
            fattn_debug_dump_output(stream, params.dst, D, params.ne01, params.ne02);

            dbg.close_file();

            fprintf(stderr, "[FA-DEBUG] FA_ON call %d: ne01=%d, ne02=%d, ne12=%d, D=%d, n_kv=%d, scale=%.4f\n",
                    fa_call_count, params.ne01, params.ne02, params.ne12, D, params.ne11, params.scale);
        }
    }
}
