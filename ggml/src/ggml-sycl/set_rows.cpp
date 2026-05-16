#include "set_rows.hpp"

#include "common.hpp"
#include "cpy.hpp"

static constexpr int GGML_SYCL_SET_ROWS_UNKNOWN_DEVICE_USM = -2;

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

static const ggml_tensor * ggml_sycl_set_rows_root(const ggml_tensor * tensor, size_t & view_offs) {
    view_offs = 0;
    if (!tensor) {
        return nullptr;
    }
    view_offs                = tensor->view_src ? tensor->view_offs : 0;
    const ggml_tensor * root = tensor;
    while (root->view_src) {
        root = root->view_src;
    }
    return root;
}

static int ggml_sycl_set_rows_owner_device(const ggml_tensor * dst, int fallback_device) {
    size_t              view_offs = 0;
    const ggml_tensor * root      = ggml_sycl_set_rows_root(dst, view_offs);
    if (!root) {
        root = dst;
    }

    if (root && root->extra) {
        const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(root->extra);
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
        return false;
    }

    if (!out.src0_ptr || !out.index_ptr || !out.dst_ptr || !dst_on_owner) {
        return false;
    }

    *plan = out;
    return true;
}

static const void * ggml_sycl_set_rows_stage_ptr(ggml_backend_sycl_context &            ctx,
                                                 const void *                           ptr,
                                                 size_t                                 bytes,
                                                 int                                    owner_device,
                                                 int                                    src_device,
                                                 std::vector<ggml_sycl::alloc_handle> & staged_allocs) {
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
    ggml_sycl::scoped_unified_alloc scoped_alloc(req);
    if (!scoped_alloc) {
        GGML_LOG_ERROR("[SYCL] SET_ROWS failed to stage %zu bytes to owner device %d\n", bytes, owner_device);
        return nullptr;
    }

    if (src_device >= 0 && src_device != owner_device) {
        queue_ptr src_stream = ctx.stream(src_device, 0);

        struct host_tmp_guard {
            void *        ptr   = nullptr;
            size_t        bytes = 0;
            sycl::queue * queue = nullptr;

            ~host_tmp_guard() {
                if (ptr && queue) {
                    ggml_sycl_free_host_tracked_bytes(ptr, bytes, *queue);
                }
            }
        } host_tmp;

        host_tmp.ptr   = ggml_sycl_malloc_host_tracked_bytes(bytes, *src_stream, "set_rows_cross_device_stage");
        host_tmp.bytes = bytes;
        host_tmp.queue = src_stream;
        if (!host_tmp.ptr) {
            GGML_LOG_ERROR("[SYCL] SET_ROWS failed to allocate host bounce for %zu bytes from device %d to %d\n", bytes,
                           src_device, owner_device);
            return nullptr;
        }
        src_stream->memcpy(host_tmp.ptr, ptr, bytes).wait_and_throw();
        stream->memcpy(scoped_alloc.get(), host_tmp.ptr, bytes).wait_and_throw();
    } else {
        stream->memcpy(scoped_alloc.get(), ptr, bytes).wait_and_throw();
    }
    ggml_sycl::alloc_handle alloc = scoped_alloc.release();
    staged_allocs.push_back(alloc);
    return alloc.ptr;
}

static void ggml_sycl_set_rows_free_staged(std::vector<ggml_sycl::alloc_handle> & staged_allocs) {
    for (auto & alloc : staged_allocs) {
        (void) ggml_sycl::unified_free(alloc);
    }
    staged_allocs.clear();
}

struct ggml_sycl_set_rows_staged_alloc_guard {
    std::vector<ggml_sycl::alloc_handle> handles;

    ~ggml_sycl_set_rows_staged_alloc_guard() { ggml_sycl_set_rows_free_staged(handles); }
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

    sycl::event evt =
        stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
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
            const size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
            const char *  src_block   = src0_d + src_offset + i00 * sizeof(float);
            const size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
            const int64_t dst_row     = src1_d[src1_offset / sizeof(TIdx)];
            const size_t  dst_offset =
                calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * sizeof(blockType);
            char * dst_block = reinterpret_cast<char *>(reinterpret_cast<char *>(dst_d) + dst_offset);
            cpyblck(src_block, dst_block);
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

    return stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        k_set_rows<TIn, TIdx, TOut>(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne11, ne12, nb01, nb02, nb03, nb10, nb11,
                                    nb12, nb1, nb2, nb3, src_type_size, dst_type_size, total_elements, item_ct1);
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

    return stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        k_set_rows_fp8<TIdx>(src0_d, src1_d, dst_d, ne00, ne01, ne02, ne11, ne12, nb01, nb02, nb03, nb10, nb11, nb12,
                             nb1, nb2, nb3, total_elements, item_ct1);
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
        float * check_buf =
            (float *) ggml_sycl_malloc_host_tracked_bytes(32 * sizeof(float), *ctx.stream(), "set_rows_debug");
        ctx.stream()->memcpy(check_buf, src0_d, 32 * sizeof(float)).wait();
        int zeros = 0;
        for (int i = 0; i < 32; i++) {
            if (check_buf[i] == 0.0f) {
                zeros++;
            }
        }
        fprintf(stderr, "  src0 data[0..4]=%.4f,%.4f,%.4f,%.4f,%.4f zeros=%d/32\n", check_buf[0], check_buf[1],
                check_buf[2], check_buf[3], check_buf[4], zeros);
        ggml_sycl_free_host_tracked_bytes(check_buf, 32 * sizeof(float), *ctx.stream());
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
        case GGML_TYPE_F8_E4M3:
            evt = set_rows_sycl_fp8<TIdx>(src0_d, src1_d, (fp8_e4m3_t *) dst_d, ne00, ne01, ne02, ne03, ne11, ne12,
                                          nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
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
    GGML_ASSERT(ggml_sycl_plan_set_rows(dst.raw(), ctx.device, &plan));

    ggml_sycl_set_rows_staged_alloc_guard staged_allocs;
    if (plan.src0_needs_staging) {
        plan.src0_ptr = ggml_sycl_set_rows_stage_ptr(ctx, plan.src0_ptr, ggml_nbytes(src0), plan.owner_device,
                                                     plan.src0_device, staged_allocs.handles);
        GGML_ASSERT(plan.src0_ptr != nullptr);
    }
    if (plan.index_needs_staging) {
        plan.index_ptr = ggml_sycl_set_rows_stage_ptr(ctx, plan.index_ptr, ggml_nbytes(src1), plan.owner_device,
                                                      plan.index_device, staged_allocs.handles);
        GGML_ASSERT(plan.index_ptr != nullptr);
    }

    sycl::event evt;
    if (src1->type == GGML_TYPE_I64) {
        evt = set_rows_sycl<float, int64_t>(ctx, src0, src1, dst, plan);
    } else {
        evt = set_rows_sycl<float, int32_t>(ctx, src0, src1, dst, plan);
    }

    // Note: KV cache synchronization is now handled via barrier-based sync in llama-context.cpp
    // The barrier approach provides ~2% better prompt eval performance than full queue sync
    if (!staged_allocs.handles.empty()) {
        evt.wait_and_throw();
    } else {
        (void) evt;
    }
}
