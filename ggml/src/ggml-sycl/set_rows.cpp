#include "set_rows.hpp"

#include "common.hpp"
#include "cpy.hpp"
#include "fattn.hpp"
#include "mem-ops.hpp"
#include "sycl-kernel-profiler.hpp"

#include <utility>

static constexpr int GGML_SYCL_SET_ROWS_UNKNOWN_DEVICE_USM = -2;

static ggml_sycl_profile_label make_set_rows_profile_label(sycl::queue & queue,
                                                           const char *  name,
                                                           const char *  metadata,
                                                           size_t        bytes = 0) {
    ggml_sycl_profile_label label{};
    label.name       = name;
    label.category   = "memory";
    label.queue_kind = "compute";
    label.metadata   = metadata;
    label.device     = ggml_sycl_get_device_id_from_queue(queue);
    label.bytes      = bytes;
    return label;
}

template <typename SubmitFn>
static sycl::event ggml_sycl_set_rows_profile_submit(sycl::queue & queue,
                                                     const char *  name,
                                                     const char *  metadata,
                                                     size_t        bytes,
                                                     SubmitFn &&   submit_fn,
                                                     const char *  file     = __builtin_FILE(),
                                                     int           line     = __builtin_LINE(),
                                                     const char *  function = __builtin_FUNCTION()) {
    if (!ggml_sycl_kernel_profile_enabled()) {
        return submit_fn(queue);
    }

    ggml_sycl_profile_label label = make_set_rows_profile_label(queue, name, metadata, bytes);
    return ggml_sycl_profile_submit(queue, label, static_cast<SubmitFn &&>(submit_fn), file, line, function);
}

static int ggml_sycl_set_rows_ptr_device(const void * ptr) {
    if (!ptr) {
        return ggml_sycl::mem_handle::HOST_DEVICE;
    }

    const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
    if (info && info->type == ggml_sycl::alloc_type::DEVICE) {
        return info->device_id;
    }

    return ggml_sycl::mem_handle::HOST_DEVICE;
}

static ggml_sycl::mem_handle ggml_sycl_set_rows_copy_handle_for_raw_ptr(void * ptr,
                                                                        int    fallback_device,
                                                                        bool   fallback_on_device) {
    return ggml_sycl_memcpy_handle_for_raw_ptr(ptr, fallback_device, GGML_LAYOUT_AOS, fallback_on_device,
                                               /*fallback_unknown=*/true);
}

static bool ggml_sycl_set_rows_alloc_host_stage(size_t                  bytes,
                                                sycl::queue &           queue,
                                                int                     device,
                                                const char *            cohort_id,
                                                bool                    require_host_usm_base,
                                                ggml_sycl::mem_handle & owner) {
    owner = {};
    if (bytes == 0) {
        return true;
    }

    ggml_sycl::alloc_request req{};
    req.queue                                    = &queue;
    req.device                                   = device;
    req.size                                     = bytes;
    req.intent.role                              = ggml_sycl::alloc_role::EXPERT_STAGING;
    req.intent.category                          = ggml_sycl::runtime_category::STAGING;
    req.intent.cohort_id                         = cohort_id;
    req.intent.constraints.must_host_pinned      = true;
    req.intent.constraints.use_pinned_pool       = true;
    req.intent.constraints.require_host_usm_base = require_host_usm_base;
    req.suppress_failure_log                     = true;

    owner = ggml_sycl::unified_allocate(req);
    if (!owner.valid()) {
        return false;
    }

    auto resolved = owner.resolve(device);
    if (!resolved || !resolved.ptr || resolved.on_device) {
        owner = {};
        return false;
    }
    return true;
}

static bool ggml_sycl_set_rows_ptr_needs_staging(const void * ptr,
                                                 int          owner_device,
                                                 bool         already_on_owner_device,
                                                 int *        ptr_device) {
    if (ptr_device) {
        *ptr_device = ggml_sycl_set_rows_ptr_device(ptr);
    }
    if (!ptr) {
        return false;
    }
    if (already_on_owner_device) {
        return false;
    }

    const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
    if (info) {
        if (info->type == ggml_sycl::alloc_type::DEVICE) {
            if (ptr_device) {
                *ptr_device = info->device_id;
            }
            return info->device_id != owner_device;
        }
        return false;
    }

    sycl::usm::alloc alloc_type = sycl::usm::alloc::unknown;
    try {
        alloc_type = ggml_sycl_get_alloc_type(ptr);
    } catch (...) {
        alloc_type = sycl::usm::alloc::unknown;
    }
    if (alloc_type == sycl::usm::alloc::unknown) {
        const int device_count = std::max(ggml_sycl_info().device_count, ggml_sycl_info().total_gpu_count);
        for (int d = 0; d < device_count && d < GGML_SYCL_MAX_DEVICES; ++d) {
            try {
                sycl::context ctx = ggml_sycl_get_device(d).default_queue().get_context();
                alloc_type        = sycl::get_pointer_type(ptr, ctx);
                if (alloc_type != sycl::usm::alloc::unknown) {
                    break;
                }
            } catch (...) {
                alloc_type = sycl::usm::alloc::unknown;
            }
        }
    }
    if (alloc_type == sycl::usm::alloc::host || alloc_type == sycl::usm::alloc::shared) {
        return false;
    }
    if (alloc_type == sycl::usm::alloc::device) {
        if (ptr_device) {
            *ptr_device = GGML_SYCL_SET_ROWS_UNKNOWN_DEVICE_USM;
        }
        return true;
    }

    return true;
}

static bool ggml_sycl_set_rows_dst_device_accessible(const void * ptr, int owner_device, bool already_on_owner_device) {
    if (!ptr) {
        return false;
    }
    if (already_on_owner_device) {
        return true;
    }

    const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
    if (info) {
        if (info->type == ggml_sycl::alloc_type::DEVICE) {
            return info->device_id == owner_device;
        }
        return info->type == ggml_sycl::alloc_type::HOST_PINNED || info->type == ggml_sycl::alloc_type::SHARED;
    }

    const sycl::usm::alloc alloc_type = ggml_sycl_probe_alloc_type_any_context(ptr);
    return alloc_type == sycl::usm::alloc::host || alloc_type == sycl::usm::alloc::shared;
}

static const ggml_tensor * ggml_sycl_set_rows_root(const ggml_tensor * tensor, size_t & view_offs) {
    return ggml_sycl_view_root_and_offset(tensor, view_offs);
}

static int ggml_sycl_set_rows_owner_device(const ggml_tensor * dst, int fallback_device) {
    size_t              view_offs = 0;
    const ggml_tensor * root      = ggml_sycl_set_rows_root(dst, view_offs);
    if (!root) {
        root = dst;
    }

    if (root && root->extra) {
        const auto * extra       = static_cast<const ggml_tensor_extra_gpu *>(root->extra);
        int          only_device = -1;
        int          n_devices   = 0;

        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            auto resolved = extra->data_handle[d].resolve(d);
            if (resolved && resolved.on_device && root->data == resolved.ptr) {
                return d;
            }
            if (resolved && resolved.on_device) {
                only_device = d;
                ++n_devices;
            }
        }
        if (n_devices == 1) {
            return only_device;
        }
        if (fallback_device >= 0 && fallback_device < GGML_SYCL_MAX_DEVICES) {
            auto resolved = extra->data_handle[fallback_device].resolve(fallback_device);
            if (resolved && resolved.on_device) {
                return fallback_device;
            }
        }
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            auto resolved = extra->data_handle[d].resolve(d);
            if (resolved && resolved.on_device) {
                return d;
            }
        }
    }

    if (root && root->data) {
        const auto * info = ggml_sycl::alloc_registry::instance().lookup(root->data);
        if (info && info->type == ggml_sycl::alloc_type::DEVICE && info->device_id >= 0 &&
            info->device_id < GGML_SYCL_MAX_DEVICES) {
            return info->device_id;
        }
    }

    return fallback_device;
}

static void * ggml_sycl_set_rows_resolve_view_on_device(const ggml_tensor * tensor, int device, bool * on_device) {
    if (on_device) {
        *on_device = false;
    }
    if (!tensor) {
        return nullptr;
    }

    size_t              view_offs = 0;
    const ggml_tensor * root      = ggml_sycl_set_rows_root(tensor, view_offs);
    if (!root) {
        root = tensor;
    }

    if (root->extra && device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        auto * extra    = static_cast<ggml_tensor_extra_gpu *>(root->extra);
        auto   resolved = extra->data_handle[device].resolve(device);
        if (resolved) {
            if (on_device) {
                *on_device = resolved.on_device;
            }
            return static_cast<char *>(resolved.ptr) + view_offs;
        }
    }

    if (!root->extra && tensor->data) {
        return const_cast<void *>(ggml_sycl_host_data(tensor));
    }

    void * ptr = ggml_sycl_resolve_tensor_ptr(tensor, device);
    if (ptr) {
        const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
        if (on_device && info && info->type == ggml_sycl::alloc_type::DEVICE && info->device_id == device) {
            *on_device = true;
        }
        return ptr;
    }

    return const_cast<void *>(ggml_sycl_host_data(tensor));
}

static void * ggml_sycl_set_rows_resolve_view_any_device(const ggml_tensor * tensor, int * source_device) {
    if (source_device) {
        *source_device = ggml_sycl::mem_handle::HOST_DEVICE;
    }
    if (!tensor) {
        return nullptr;
    }

    size_t              view_offs = 0;
    const ggml_tensor * root      = ggml_sycl_set_rows_root(tensor, view_offs);
    if (!root) {
        root = tensor;
    }

    if (root->extra) {
        auto * extra = static_cast<ggml_tensor_extra_gpu *>(root->extra);
        for (int d = 0; d < GGML_SYCL_MAX_DEVICES; ++d) {
            auto resolved = extra->data_handle[d].resolve(d);
            if (resolved && resolved.on_device) {
                if (source_device) {
                    *source_device = d;
                }
                return static_cast<char *>(resolved.ptr) + view_offs;
            }
        }
    }

    return const_cast<void *>(ggml_sycl_host_data(tensor));
}

static const char * ggml_sycl_set_rows_alloc_type_name(const void * ptr) {
    if (!ptr) {
        return "null";
    }
    const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
    if (!info) {
        return "unregistered";
    }
    switch (info->type) {
        case ggml_sycl::alloc_type::DEVICE:
            return "device";
        case ggml_sycl::alloc_type::HOST_PINNED:
            return "host_pinned";
        case ggml_sycl::alloc_type::SHARED:
            return "shared";
        case ggml_sycl::alloc_type::MMAP:
            return "mmap";
        default:
            return "other";
    }
}

static int ggml_sycl_set_rows_alloc_device(const void * ptr) {
    if (!ptr) {
        return ggml_sycl::mem_handle::HOST_DEVICE;
    }
    const auto * info = ggml_sycl::alloc_registry::instance().lookup(ptr);
    return info ? info->device_id : ggml_sycl::mem_handle::HOST_DEVICE;
}

static void ggml_sycl_set_rows_log_tensor_resolution(const char * role, const ggml_tensor * tensor, int device) {
    if (!tensor) {
        GGML_LOG_ERROR("[SYCL] SET_ROWS plan fail %s: null tensor\n", role);
        return;
    }

    size_t              view_offs = 0;
    const ggml_tensor * root      = ggml_sycl_set_rows_root(tensor, view_offs);
    if (!root) {
        root = tensor;
    }

    bool   on_owner = false;
    void * ptr      = ggml_sycl_set_rows_resolve_view_on_device(tensor, device, &on_owner);
    GGML_LOG_ERROR(
        "[SYCL] SET_ROWS plan fail %s: tensor=%s op=%s type=%s data=%p view_src=%s view_offs=%zu root=%s "
        "root_data=%p root_extra=%p resolved=%p on_owner=%d alloc=%s alloc_dev=%d\n",
        role, tensor->name ? tensor->name : "?", ggml_op_name(tensor->op), ggml_type_name(tensor->type), tensor->data,
        tensor->view_src ? (tensor->view_src->name ? tensor->view_src->name : "?") : "none", view_offs,
        root->name ? root->name : "?", root->data, root->extra, ptr, on_owner ? 1 : 0,
        ggml_sycl_set_rows_alloc_type_name(ptr), ggml_sycl_set_rows_alloc_device(ptr));

    if (root->extra && device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        auto * extra = static_cast<ggml_tensor_extra_gpu *>(root->extra);
        for (int d = 0; d < ggml_sycl_info().device_count && d < GGML_SYCL_MAX_DEVICES; ++d) {
            auto resolved = extra->data_handle[d].resolve(d);
            GGML_LOG_ERROR(
                "[SYCL] SET_ROWS plan fail %s root slot %d: raw=%p handle_ptr=%p handle_on_device=%d "
                "handle_valid=%d alloc=%s alloc_dev=%d\n",
                role, d, extra->data_device[d], resolved.ptr, resolved.on_device ? 1 : 0,
                extra->data_handle[d].valid() ? 1 : 0, ggml_sycl_set_rows_alloc_type_name(resolved.ptr),
                ggml_sycl_set_rows_alloc_device(resolved.ptr));
        }
    }
}

static void ggml_sycl_set_rows_log_plan_failure(const ggml_tensor *             dst,
                                                const ggml_sycl_set_rows_plan & out,
                                                bool                            src0_on_owner,
                                                bool                            index_on_owner,
                                                bool                            dst_on_owner,
                                                bool                            dst_device_accessible) {
    static std::atomic<int> log_count{ 0 };
    if (log_count.fetch_add(1, std::memory_order_relaxed) >= 8) {
        return;
    }

    GGML_LOG_ERROR(
        "[SYCL] SET_ROWS plan failed: owner=%d src0=%p src0_on_owner=%d src0_stage=%d src0_device=%d "
        "index=%p index_on_owner=%d index_stage=%d index_device=%d dst=%p dst_on_owner=%d dst_accessible=%d\n",
        out.owner_device, out.src0_ptr, src0_on_owner ? 1 : 0, out.src0_needs_staging ? 1 : 0, out.src0_device,
        out.index_ptr, index_on_owner ? 1 : 0, out.index_needs_staging ? 1 : 0, out.index_device, out.dst_ptr,
        dst_on_owner ? 1 : 0, dst_device_accessible ? 1 : 0);
    ggml_sycl_set_rows_log_tensor_resolution("src0", dst ? dst->src[0] : nullptr, out.owner_device);
    ggml_sycl_set_rows_log_tensor_resolution("index", dst ? dst->src[1] : nullptr, out.owner_device);
    ggml_sycl_set_rows_log_tensor_resolution("dst", dst, out.owner_device);
}

bool ggml_sycl_plan_set_rows(const ggml_tensor * dst, int fallback_device, ggml_sycl_set_rows_plan * plan) {
    if (!dst || !dst->src[0] || !dst->src[1] || !plan) {
        return false;
    }

    ggml_sycl_set_rows_plan out{};
    out.owner_device = ggml_sycl_set_rows_owner_device(dst, fallback_device);

    bool src0_on_owner  = false;
    bool index_on_owner = false;
    bool dst_on_owner   = false;
    out.src0_ptr        = ggml_sycl_set_rows_resolve_view_on_device(dst->src[0], out.owner_device, &src0_on_owner);
    out.index_ptr       = ggml_sycl_set_rows_resolve_view_on_device(dst->src[1], out.owner_device, &index_on_owner);
    out.dst_ptr         = ggml_sycl_set_rows_resolve_view_on_device(dst, out.owner_device, &dst_on_owner);

    int    src0_any_device  = ggml_sycl::mem_handle::HOST_DEVICE;
    int    index_any_device = ggml_sycl::mem_handle::HOST_DEVICE;
    void * src0_any_ptr     = ggml_sycl_set_rows_resolve_view_any_device(dst->src[0], &src0_any_device);
    void * index_any_ptr    = ggml_sycl_set_rows_resolve_view_any_device(dst->src[1], &index_any_device);
    if (!src0_on_owner && src0_any_device >= 0 && src0_any_ptr) {
        out.src0_ptr = src0_any_ptr;
    } else if (!out.src0_ptr) {
        out.src0_ptr = src0_any_ptr;
    }
    if (!index_on_owner && index_any_device >= 0 && index_any_ptr) {
        out.index_ptr = index_any_ptr;
    } else if (!out.index_ptr) {
        out.index_ptr = index_any_ptr;
    }

    out.src0_needs_staging =
        ggml_sycl_set_rows_ptr_needs_staging(out.src0_ptr, out.owner_device, src0_on_owner, &out.src0_device);
    out.index_needs_staging =
        ggml_sycl_set_rows_ptr_needs_staging(out.index_ptr, out.owner_device, index_on_owner, &out.index_device);
    if (out.src0_device == ggml_sycl::mem_handle::HOST_DEVICE && src0_any_device >= 0) {
        out.src0_device = src0_any_device;
    }
    if (out.index_device == ggml_sycl::mem_handle::HOST_DEVICE && index_any_device >= 0) {
        out.index_device = index_any_device;
    }
    if ((out.src0_needs_staging && out.src0_device == GGML_SYCL_SET_ROWS_UNKNOWN_DEVICE_USM) ||
        (out.index_needs_staging && out.index_device == GGML_SYCL_SET_ROWS_UNKNOWN_DEVICE_USM)) {
        ggml_sycl_set_rows_log_plan_failure(dst, out, src0_on_owner, index_on_owner, dst_on_owner,
                                            /*dst_device_accessible=*/false);
        return false;
    }

    const bool dst_device_accessible =
        ggml_sycl_set_rows_dst_device_accessible(out.dst_ptr, out.owner_device, dst_on_owner);
    if (!out.src0_ptr || !out.index_ptr || !out.dst_ptr || !dst_device_accessible) {
        ggml_sycl_set_rows_log_plan_failure(dst, out, src0_on_owner, index_on_owner, dst_on_owner,
                                            dst_device_accessible);
        return false;
    }

    *plan = out;
    return true;
}

static const void * ggml_sycl_set_rows_stage_ptr(ggml_backend_sycl_context &          ctx,
                                                 const void *                         ptr,
                                                 size_t                               bytes,
                                                 int                                  owner_device,
                                                 int                                  src_device,
                                                 std::vector<ggml_sycl::mem_handle> & staged_handles) {
    if (!ptr || bytes == 0) {
        return ptr;
    }

    queue_ptr stream = ctx.stream(owner_device, 0);

    ggml_sycl::alloc_request req;
    req.queue                          = stream;
    req.device                         = owner_device;
    req.size                           = bytes;
    req.intent.role                    = ggml_sycl::alloc_role::GRAPH_TMP;
    req.intent.category                = ggml_sycl::runtime_category::GRAPH;
    req.intent.constraints.must_device = true;
    ggml_sycl::mem_handle staged_handle = ggml_sycl::unified_allocate(req);
    auto                  staged_res    = staged_handle.resolve(owner_device);
    if (!staged_res || !staged_res.on_device || !staged_res.ptr) {
        GGML_LOG_ERROR("[SYCL] SET_ROWS failed to stage %zu bytes to owner device %d\n", bytes, owner_device);
        return nullptr;
    }

    if (src_device >= 0 && src_device != owner_device) {
        queue_ptr src_stream = ctx.stream(src_device, 0);

        ggml_sycl::mem_handle host_handle;
        if (!ggml_sycl_set_rows_alloc_host_stage(bytes, *src_stream, src_device, "set_rows_cross_device_stage",
                                                 /*require_host_usm_base=*/true, host_handle)) {
            GGML_LOG_ERROR("[SYCL] SET_ROWS failed to allocate host bounce for %zu bytes from device %d to %d\n", bytes,
                           src_device, owner_device);
            return nullptr;
        }
        auto resolved_host = host_handle.resolve(src_device);
        GGML_ASSERT(resolved_host && !resolved_host.on_device);

        auto src_handle = ggml_sycl_set_rows_copy_handle_for_raw_ptr(const_cast<void *>(ptr), src_device, true);
        ggml_sycl::mem_copy(host_handle, src_handle, bytes, *src_stream);

        ggml_sycl::mem_copy(staged_handle, host_handle, bytes, *stream);
    } else {
        auto src_handle =
            ggml_sycl_set_rows_copy_handle_for_raw_ptr(const_cast<void *>(ptr), owner_device, src_device >= 0);
        ggml_sycl::mem_copy(staged_handle, src_handle, bytes, *stream);
    }
    const void * staged = staged_res.ptr;
    staged_handles.push_back(std::move(staged_handle));
    return staged;
}

struct ggml_sycl_set_rows_staged_handle_guard {
    std::vector<ggml_sycl::mem_handle> handles;
};

namespace utils {
template <typename T> static constexpr bool is_arithmetic_v() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half> || std::is_same_v<T, sycl::ext::oneapi::bfloat16>;
}
}  // namespace utils

// FP8 E4M3 type for SYCL (matches ggml_fp8_e4m3_t)
struct fp8_e4m3_t {
    uint8_t bits;
};

// Device-side FP32 to FP8 E4M3 conversion
// E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Bias: 7, range: ±448, no infinity
inline fp8_e4m3_t fp32_to_fp8_e4m3(float x) {
    fp8_e4m3_t result;

    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = x;

    uint32_t sign = (bits.u >> 31) & 0x1;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127;  // unbias FP32 exponent
    uint32_t mant = bits.u & 0x7FFFFF;

    // Handle special cases
    if (exp == 128) {                      // NaN or Inf in FP32
        result.bits = 0x7F | (sign << 7);  // E4M3 NaN (max value with sign)
        return result;
    }

    if (exp < -9) {  // Too small, underflow to zero
        result.bits = sign << 7;
        return result;
    }

    // Rebias for E4M3 (bias = 7)
    exp += 7;

    if (exp <= 0) {  // Subnormal in E4M3
        // Shift mantissa to create subnormal
        mant = (mant | 0x800000) >> (1 - exp);
        exp  = 0;
    } else if (exp >= 15) {                // Overflow to max value (no inf in E4M3)
        result.bits = 0x7E | (sign << 7);  // Max normal value (exp=14, mant=7)
        return result;
    }

    // Round to nearest even for mantissa (23 bits -> 3 bits)
    uint32_t mant3 = (mant + 0x100000) >> 20;  // Round and shift
    if (mant3 > 7) {
        mant3 = 0;
        exp++;
        if (exp >= 15) {
            result.bits = 0x7E | (sign << 7);  // Max normal value
            return result;
        }
    }

    result.bits = (sign << 7) | (exp << 3) | mant3;
    return result;
}

template <typename TIn, typename TOut>
static inline std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), void> convert(
    const char * src,
    char *       dst) {
    auto src_val = *reinterpret_cast<const TIn *>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<TOut *>(dst) = dst_val;
}

template <typename TIdx, typename blockType, int qk, cpy_kernel_t cpyblck>
static sycl::event set_rows_sycl_q(const char * __restrict__ src0_d,
                                   const TIdx * __restrict__ src1_d,
                                   blockType * __restrict__ dst_d,
                                   // tensor dimensions src0 and src1
                                   const int64_t ne00,
                                   const int64_t ne01,
                                   const int64_t ne02,
                                   const int64_t ne03,
                                   const int64_t ne10,
                                   const int64_t ne11,
                                   const int64_t ne12,
                                   const int64_t ne13,
                                   // strides for src0
                                   const size_t  nb00,
                                   const size_t  nb01,
                                   const size_t  nb02,
                                   const size_t  nb03,
                                   // strides for src1
                                   const size_t  nb10,
                                   const size_t  nb11,
                                   const size_t  nb12,
                                   const size_t  nb13,
                                   // strides for dst
                                   const size_t  nb1,
                                   const size_t  nb2,
                                   const size_t  nb3,
                                   queue_ptr     stream) {
    const int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr int block_size   = 256;
    const int64_t grid_size    = ceil_div(total_blocks, block_size);

    sycl::event evt = ggml_sycl_set_rows_profile_submit(
        *stream, "sycl.set_rows.quantized", "role=set_rows;kind=quantized",
        static_cast<size_t>(total_blocks) * sizeof(blockType), [&](sycl::queue & profiled_queue) {
            return profiled_queue.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                                               [=](sycl::nd_item<1> item_ct1) {
                                                   const int64_t i = item_ct1.get_global_linear_id();
                                                   if (i >= total_blocks) {
                                                       return;
                                                   }
                                                   const int64_t i_base      = i * qk;
                                                   const int64_t i03         = i_base / (ne00 * ne01 * ne02);
                                                   const int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
                                                   const int64_t i02         = rem1 / (ne00 * ne01);
                                                   const int64_t rem2        = rem1 - i02 * ne00 * ne01;
                                                   const int64_t i01         = rem2 / ne00;
                                                   const int64_t i00         = rem2 - i01 * ne00;
                                                   const int64_t i12         = i03 % ne12;
                                                   const int64_t i11         = i02 % ne11;
                                                   const int64_t i10         = i01;
                                                   const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 },
                                                                                                  { i01, i02, i03 });
                                                   const char * src_block = src0_d + src_offset + i00 * sizeof(float);
                                                   const size_t src1_offset =
                                                       calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                                                   const int64_t dst_row = src1_d[src1_offset / sizeof(TIdx)];
                                                   const size_t  dst_offset =
                                                       calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) +
                                                       (i00 / qk) * sizeof(blockType);
                                                   char * dst_block = reinterpret_cast<char *>(
                                                       reinterpret_cast<char *>(dst_d) + dst_offset);
                                                   cpyblck(src_block, dst_block);
                                               });
        });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
    return evt;
}

template <typename TIn, typename TIdx, typename TOut>
static void k_set_rows(const char * __restrict__ src0,
                       const TIdx * __restrict__ src1,
                       char * __restrict__ dst,
                       const int64_t            ne00,
                       const int64_t            ne01,
                       const int64_t            ne02,
                       const int64_t            ne11,
                       const int64_t            ne12,
                       const size_t             nb01,
                       const size_t             nb02,
                       const size_t             nb03,
                       const size_t             nb10,
                       const size_t             nb11,
                       const size_t             nb12,
                       const size_t             nb1,
                       const size_t             nb2,
                       const size_t             nb3,
                       const size_t             src_type_size,
                       const size_t             dst_type_size,
                       const int64_t            total_elements,
                       const sycl::nd_item<1> & item_ct1) {
    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row =
        *(const TIdx *) ((const char *) src1 + calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 }));

    const char * src0_row    = src0 + calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
    const char * src_elem    = src0_row + i00 * src_type_size;
    char *       dst_row_ptr = dst + dst_row * nb1 + i02 * nb2 + i03 * nb3;
    char *       dst_elem    = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template <typename TIn, typename TIdx, typename TOut>
static sycl::event set_rows_sycl(const char *  src0_d,
                                 const TIdx *  src1_d,
                                 char *        dst_d,
                                 const int64_t ne00,
                                 const int64_t ne01,
                                 const int64_t ne02,
                                 const int64_t ne03,
                                 const int64_t ne11,
                                 const int64_t ne12,
                                 const size_t  nb01,
                                 const size_t  nb02,
                                 const size_t  nb03,
                                 const size_t  nb10,
                                 const size_t  nb11,
                                 const size_t  nb12,
                                 const size_t  nb1,
                                 const size_t  nb2,
                                 const size_t  nb3,
                                 const size_t  src_type_size,
                                 const size_t  dst_type_size,
                                 queue_ptr     stream) {
    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size  = ceil_div(total_elements, block_size);

    return ggml_sycl_set_rows_profile_submit(
        *stream, "sycl.set_rows.generic", "role=set_rows;kind=generic",
        static_cast<size_t>(total_elements) * dst_type_size, [&](sycl::queue & profiled_queue) {
            return profiled_queue.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                                               [=](sycl::nd_item<1> item_ct1) {
                                                   k_set_rows<TIn, TIdx, TOut>(
                                                       src0_d, src1_d, dst_d, ne00, ne01, ne02, ne11, ne12, nb01, nb02,
                                                       nb03, nb10, nb11, nb12, nb1, nb2, nb3, src_type_size,
                                                       dst_type_size, total_elements, item_ct1);
                                               });
        });
}

// FP8 E4M3 specific kernel (can't use templated convert with non-SYCL type)
template <typename TIdx>
static void k_set_rows_fp8(const char * __restrict__ src0,
                           const TIdx * __restrict__ src1,
                           fp8_e4m3_t * __restrict__ dst,
                           const int64_t            ne00,
                           const int64_t            ne01,
                           const int64_t            ne02,
                           const int64_t            ne11,
                           const int64_t            ne12,
                           const size_t             nb01,
                           const size_t             nb02,
                           const size_t             nb03,
                           const size_t             nb10,
                           const size_t             nb11,
                           const size_t             nb12,
                           const size_t             nb1,
                           const size_t             nb2,
                           const size_t             nb3,
                           const int64_t            total_elements,
                           const sycl::nd_item<1> & item_ct1) {
    const int64_t i = item_ct1.get_global_linear_id();
    if (i >= total_elements) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row =
        *(const TIdx *) ((const char *) src1 + calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 }));

    const char *  src0_row = src0 + calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
    const float * src_elem = (const float *) (src0_row + i00 * sizeof(float));

    // Calculate destination offset (nb1/nb2/nb3 are in bytes)
    fp8_e4m3_t * dst_row_ptr = (fp8_e4m3_t *) ((char *) dst + dst_row * nb1 + i02 * nb2 + i03 * nb3);
    dst_row_ptr[i00]         = fp32_to_fp8_e4m3(*src_elem);
}

template <typename TIdx>
static sycl::event set_rows_sycl_fp8(const char *  src0_d,
                                     const TIdx *  src1_d,
                                     fp8_e4m3_t *  dst_d,
                                     const int64_t ne00,
                                     const int64_t ne01,
                                     const int64_t ne02,
                                     const int64_t ne03,
                                     const int64_t ne11,
                                     const int64_t ne12,
                                     const size_t  nb01,
                                     const size_t  nb02,
                                     const size_t  nb03,
                                     const size_t  nb10,
                                     const size_t  nb11,
                                     const size_t  nb12,
                                     const size_t  nb1,
                                     const size_t  nb2,
                                     const size_t  nb3,
                                     queue_ptr     stream) {
    const int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr int block_size = 64;
    const int64_t grid_size  = ceil_div(total_elements, block_size);

    return ggml_sycl_set_rows_profile_submit(
        *stream, "sycl.set_rows.fp8", "role=set_rows;kind=fp8",
        static_cast<size_t>(total_elements) * sizeof(fp8_e4m3_t), [&](sycl::queue & profiled_queue) {
            return profiled_queue.parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                                               [=](sycl::nd_item<1> item_ct1) {
                                                   k_set_rows_fp8<TIdx>(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne11,
                                                                        ne12, nb01, nb02, nb03, nb10, nb11, nb12, nb1,
                                                                        nb2, nb3, total_elements, item_ct1);
                                               });
        });
}

template <typename TIn, typename TIdx>
static sycl::event set_rows_sycl(ggml_backend_sycl_context &     ctx,
                                 const ggml_tensor *             src0,
                                 const ggml_tensor *             src1,
                                 ggml_sycl::sycl_tensor          safe_dst,
                                 const ggml_sycl_set_rows_plan & plan) {
    const ggml_tensor * dst_raw = safe_dst.raw();
    const ggml_tensor * dst     = dst_raw;
    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int           device  = plan.owner_device;
    const char *        src0_d  = (const char *) plan.src0_ptr;
    const TIdx *        src1_d  = (const TIdx *) plan.index_ptr;
    char *              dst_d   = (char *) plan.dst_ptr;

    // Debug: Check set_rows in TP mode
    static int set_rows_debug = 0;
    if (g_ggml_sycl_tp_debug && g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 && set_rows_debug < 6) {
        fprintf(stderr, "SYCL TP SET_ROWS[%d] dev=%d dst=%s type=%d\n", set_rows_debug, device,
                dst_raw->name ? dst_raw->name : "(null)", (int) dst_raw->type);
        fprintf(stderr, "  src0=%p->%p (ne=[%ld,%ld,%ld,%ld])\n", const_cast<void *>(ggml_sycl_host_data(src0)),
                (void *) src0_d, (long) src0->ne[0], (long) src0->ne[1], (long) src0->ne[2], (long) src0->ne[3]);
        fprintf(stderr, "  dst=%p->%p (ne=[%ld,%ld,%ld,%ld])\n", const_cast<void *>(ggml_sycl_host_data(dst_raw)),
                (void *) dst_d, (long) dst_raw->ne[0], (long) dst_raw->ne[1], (long) dst_raw->ne[2],
                (long) dst_raw->ne[3]);

        // Check if dst is a view and trace to parent
        if (dst_raw->view_src != nullptr) {
            fprintf(stderr, "  dst is VIEW of parent=%s data=%p\n", dst_raw->view_src->name,
                    const_cast<void *>(ggml_sycl_host_data(dst_raw->view_src)));
            if (dst_raw->view_src->extra) {
                const auto * parent_extra = static_cast<const ggml_tensor_extra_gpu *>(dst_raw->view_src->extra);
                fprintf(stderr, "  parent extra: data_device[0]=%p data_device[1]=%p\n", parent_extra->data_device[0],
                        parent_extra->data_device[1]);
            } else {
                fprintf(stderr, "  parent extra: NULL\n");
            }
        } else {
            fprintf(stderr, "  dst is NOT a view (view_src=NULL)\n");
        }
        if (dst_raw->extra) {
            const auto * dst_extra = static_cast<const ggml_tensor_extra_gpu *>(dst_raw->extra);
            fprintf(stderr, "  dst extra: data_device[0]=%p data_device[1]=%p\n", dst_extra->data_device[0],
                    dst_extra->data_device[1]);
        } else {
            fprintf(stderr, "  dst extra: NULL\n");
        }

        // Check src0 first values
        ggml_sycl::mem_handle check_handle;
        dpct::queue_ptr       debug_stream = ctx.stream(device, 0);
        if (ggml_sycl_set_rows_alloc_host_stage(32 * sizeof(float), *debug_stream, device, "set_rows_debug",
                                                /*require_host_usm_base=*/false, check_handle)) {
            auto                  resolved_check = check_handle.resolve(device);
            float *               check_buf      = static_cast<float *>(resolved_check.ptr);
            ggml_sycl::mem_handle src0_handle =
                ggml_sycl_set_rows_copy_handle_for_raw_ptr(const_cast<char *>(src0_d), device, true);
            ggml_sycl::mem_copy(check_handle, 0, src0_handle, 0, 32 * sizeof(float), *debug_stream);
            int zeros = 0;
            for (int i = 0; i < 32; i++) {
                if (check_buf[i] == 0.0f) {
                    zeros++;
                }
            }
            fprintf(stderr, "  src0 data[0..4]=%.4f,%.4f,%.4f,%.4f,%.4f zeros=%d/32\n", check_buf[0], check_buf[1],
                    check_buf[2], check_buf[3], check_buf[4], zeros);
        } else {
            fprintf(stderr, "  src0 debug readback allocation failed\n");
        }
        set_rows_debug++;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream(device, 0);
    sycl::event     evt;

    switch (dst_raw->type) {
        case GGML_TYPE_F32:
            evt = set_rows_sycl<TIn, TIdx, float>(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, ne11, ne12, nb01, nb02,
                                                  nb03, nb10, nb11, nb12, nb1, nb2, nb3, sizeof(TIn), sizeof(float),
                                                  stream);
            break;
        case GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            evt = set_rows_sycl<TIn, TIdx, sycl::half>(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, ne11, ne12, nb01,
                                                       nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, sizeof(TIn),
                                                       sizeof(sycl::half), stream);
            break;
        case GGML_TYPE_BF16:
            evt = set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, dst_d, ne00, ne01, ne02, ne03, ne11, ne12, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2,
                nb3, sizeof(TIn), sizeof(sycl::ext::oneapi::bfloat16), stream);
            break;
        case GGML_TYPE_Q8_0:
            evt = set_rows_sycl_q<TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02,
                nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            evt = set_rows_sycl_q<TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02,
                nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            evt = set_rows_sycl_q<TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02,
                nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            evt = set_rows_sycl_q<TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02,
                nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            evt = set_rows_sycl_q<TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01, nb02,
                nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            evt = set_rows_sycl_q<TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst_d, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            break;

        default:
            GGML_ABORT("Unsupported tensor type!");
            break;
    }

    return evt;
}

void ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst.raw(), /*num_src=*/2);
    const ggml_tensor *  src0 = dst.src(0).raw();
    const ggml_tensor *  src1 = dst.src(1).raw();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    ggml_sycl_set_rows_plan plan{};
    const bool              plan_ok = ggml_sycl_plan_set_rows(dst.raw(), ctx.device, &plan);
    if (!plan_ok) {
        fprintf(stderr,
                "[SYCL] SET_ROWS unresolved op dst=%s ctx_device=%d owner=%d src0=%s src0_ptr=%p src0_stage=%d "
                "src0_device=%d index=%s index_ptr=%p index_stage=%d index_device=%d dst_ptr=%p\n",
                dst.raw()->name, ctx.device, plan.owner_device, src0->name, plan.src0_ptr,
                plan.src0_needs_staging ? 1 : 0, plan.src0_device, src1->name, plan.index_ptr,
                plan.index_needs_staging ? 1 : 0, plan.index_device, plan.dst_ptr);
    }
    GGML_ASSERT(plan_ok);

    ggml_sycl_set_rows_staged_handle_guard staged_handles;
    if (plan.src0_needs_staging) {
        plan.src0_ptr = ggml_sycl_set_rows_stage_ptr(ctx, plan.src0_ptr, ggml_nbytes(src0), plan.owner_device,
                                                     plan.src0_device, staged_handles.handles);
        GGML_ASSERT(plan.src0_ptr != nullptr);
    }
    if (plan.index_needs_staging) {
        plan.index_ptr = ggml_sycl_set_rows_stage_ptr(ctx, plan.index_ptr, ggml_nbytes(src1), plan.owner_device,
                                                      plan.index_device, staged_handles.handles);
        GGML_ASSERT(plan.index_ptr != nullptr);
    }

    sycl::event evt;
    if (src1->type == GGML_TYPE_I64) {
        evt = set_rows_sycl<float, int64_t>(ctx, src0, src1, dst, plan);
    } else {
        evt = set_rows_sycl<float, int32_t>(ctx, src0, src1, dst, plan);
    }

    sycl::event packed_k_evt;
    if (ggml_sycl_fattn_xmx_update_packed_k_from_set_rows(dst.raw(), src0, src1, plan.owner_device, plan.src0_ptr,
                                                          plan.index_ptr, ctx.stream(plan.owner_device, 0), evt,
                                                          &packed_k_evt)) {
        evt = packed_k_evt;
    }

    // Note: KV cache synchronization is now handled via barrier-based sync in llama-context.cpp
    // The barrier approach provides ~2% better prompt eval performance than full queue sync
    if (!staged_handles.handles.empty()) {
        evt.wait_and_throw();
    } else {
        (void) evt;
    }
}
