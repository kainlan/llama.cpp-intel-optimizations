#include "roll.hpp"

#include "common.hpp"
#include "mem-ops.hpp"

using namespace sycl;

static ggml_sycl::mem_handle roll_tensor_handle_for_copy(const ggml_sycl::sycl_tensor & tensor) {
    void * ptr = tensor.resolve_ptr();
    if (!ptr) {
        return {};
    }

    const int device = tensor.device();
    if (device >= 0 && device < GGML_SYCL_MAX_DEVICES) {
        if (auto * extra = tensor.extra_gpu()) {
            const auto & handle = extra->data_handle[device];
            if (handle.valid()) {
                auto resolved = handle.resolve(device);
                if (resolved && resolved.ptr == ptr) {
                    return handle;
                }
            }
        }
    }

    const bool on_device = ggml_sycl_get_alloc_type(ptr) == sycl::usm::alloc::device;
    return ggml_sycl::mem_handle::from_chunk_ptr(ptr, device, GGML_LAYOUT_AOS, on_device);
}

static inline int wrap_add(int i, int shift, int n) {
    int s = i + shift;
    return (s >= n) ? (s - n) : s;
}

static void kernel_roll_fused_i0_i1(queue &       q,
                                    const float * src_d,
                                    float *       dst_d,
                                    int           ne0,
                                    int           ne1,
                                    int           ne2,
                                    int           ne3,
                                    int           sh0,
                                    int           sh1,
                                    int           sh2,
                                    int           sh3) {
    if (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne3 == 0) {
        return;
    }

    const int stride1 = ne0;
    const int stride2 = ne0 * ne1;
    const int stride3 = ne0 * ne1 * ne2;

    const int shNe0 = (ne0 - sh0) % ne0;
    const int shNe1 = (ne1 - sh1) % ne1;
    const int shNe2 = (ne2 - sh2) % ne2;
    const int shNe3 = (ne3 - sh3) % ne3;

    const size_t g0 = (size_t) ne3;
    const size_t g1 = (size_t) ne2;
    const size_t g2 = (size_t) (ne1 * ne0);

    const range<3> global{ g0, g1, g2 };

    q.submit([&](handler & h) {
        h.parallel_for(global, [=](id<3> idx) {
            const int i3 = (int) idx[0];
            const int i2 = (int) idx[1];

            const int fused = (int) idx[2];
            const int i1    = fused / ne0;
            const int i0    = fused - i1 * ne0;  // fused % ne0

            const int idx_dst = i0 + i1 * stride1 + i2 * stride2 + i3 * stride3;

            const int s0 = wrap_add(i0, shNe0, ne0);
            const int s1 = wrap_add(i1, shNe1, ne1);
            const int s2 = wrap_add(i2, shNe2, ne2);
            const int s3 = wrap_add(i3, shNe3, ne3);

            const int idx_src = s0 + s1 * stride1 + s2 * stride2 + s3 * stride3;

            dst_d[idx_dst] = src_d[idx_src];
        });
    });
}

void ggml_sycl_roll(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    GGML_ASSERT(dst.type() == GGML_TYPE_F32);

    auto src = dst.src(0);
    GGML_ASSERT(src && src.type() == GGML_TYPE_F32);

    const int ne0 = (int) dst.ne(0);
    const int ne1 = (int) dst.ne(1);
    const int ne2 = (int) dst.ne(2);
    const int ne3 = (int) dst.ne(3);

    const int32_t * params = static_cast<const int32_t *>(dst.op_params());
    int             shift0 = params[0];
    int             shift1 = params[1];
    int             shift2 = params[2];
    int             shift3 = params[3];

    if ((shift0 | shift1 | shift2 | shift3) == 0) {
        const size_t nb         = src.nbytes();
        queue *      q          = ctx.stream();
        auto         dst_handle = roll_tensor_handle_for_copy(dst);
        auto         src_handle = roll_tensor_handle_for_copy(src);
        GGML_ASSERT(dst_handle.valid() && src_handle.valid());
        (void) ggml_sycl::mem_copy_async(dst_handle, src_handle, nb, *q);
        return;
    }

    auto norm = [](int sh, int n) -> int {
        if (n <= 0) {
            return 0;
        }
        sh %= n;
        if (sh < 0) {
            sh += n;
        }
        return sh;
    };
    shift0 = norm(shift0, ne0);
    shift1 = norm(shift1, ne1);
    shift2 = norm(shift2, ne2);
    shift3 = norm(shift3, ne3);

    try {
        queue * q = ctx.stream();

        const float * src_d = src.resolve_as<const float>();
        float *       dst_d = dst.resolve_as<float>();
        GGML_ASSERT(src_d && dst_d);

        kernel_roll_fused_i0_i1(*q, src_d, dst_d, ne0, ne1, ne2, ne3, shift0, shift1, shift2, shift3);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[SYCL-ROLL] ERROR: %s\n", e.what());
        throw;
    }
}
