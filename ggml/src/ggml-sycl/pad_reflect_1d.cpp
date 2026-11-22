#include "pad_reflect_1d.hpp"

#include <climits>

static void pad_reflect_1d_kernel_f32(const void * __restrict__ src0,
                                      void * __restrict__ dst,
                                      const int64_t     ne0,
                                      const int64_t     ne00,
                                      const sycl::uint3 ne01,
                                      const int64_t     ne02,
                                      const int64_t     ne03,
                                      const int64_t     nb00,
                                      const int64_t     nb01,
                                      const int64_t     nb02,
                                      const int64_t     nb03,
                                      const int64_t     nb0,
                                      const int64_t     nb1,
                                      const int64_t     nb2,
                                      const int64_t     nb3,
                                      const int         p0,
                                      const int         p1,
                                      const int64_t     n_groups0,
                                      sycl::nd_item<3>  item_ct1) {
    const int64_t i3 = item_ct1.get_group(0);
    const int64_t i2 = item_ct1.get_group(1);
    if (i2 >= ne02 || i3 >= ne03) {
        return;
    }

    // The launcher caps dimension 2. A work-group serves every row tile congruent to its own index.
    // fast_div_modulo takes a uint32_t: a tile count above 2^32 needs a tensor larger than either card.
    for (int64_t g = item_ct1.get_group(2); g < n_groups0; g += (int64_t) item_ct1.get_group_range(2)) {
        const sycl::uint2 div_mod_packed = fast_div_modulo((uint32_t) g, ne01);
        const int64_t     i1             = div_mod_packed.y();
        const int64_t     i0             = item_ct1.get_local_id(2) + div_mod_packed.x() * item_ct1.get_local_range(2);
        if (i0 >= ne0 || i1 >= ne01.z()) {
            continue;
        }

        const char * src0_ptr = (const char *) src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
        char *       dst_ptr  = (char *) dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

        const int64_t rel_i0 = i0 - p0;  // relative i0 in src0
        int64_t       src_idx;

        if (rel_i0 < 0) {
            // Left padding - reflect
            src_idx = -rel_i0;
        } else if (rel_i0 < ne00) {
            // Middle - copy
            src_idx = rel_i0;
        } else {
            // Right padding - reflect
            src_idx = 2 * ne00 - 2 - rel_i0;
        }
        const float value               = *(const float *) (src0_ptr + src_idx * nb00);
        *(float *) (dst_ptr + i0 * nb0) = value;
    }

    GGML_UNUSED(p1);
}

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context & ctx, ggml_sycl::sycl_tensor dst) {
    const ggml_tensor * dst_raw = dst.raw();
    auto src0 = dst.src(0);

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0.type() == GGML_TYPE_F32);
    GGML_ASSERT(dst.type() == GGML_TYPE_F32);

    const int32_t * opts = static_cast<const int32_t *>(dst.op_params());
    const int p0 = opts[0];
    const int p1 = opts[1];

    const int64_t ne00 = src0.ne(0);
    const int64_t ne01 = src0.ne(1);
    const sycl::uint3 ne01_packed = init_fastdiv_values(ne01);
    const int64_t ne02 = src0.ne(2);
    const int64_t ne03 = src0.ne(3);

    const int64_t ne0 = dst.ne(0);

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1D_BLOCK_SIZE;
    const int64_t tiles0 = (ne0 + bx - 1) / bx;
    const int64_t        n_groups0   = ne01 * tiles0;
    // The runtime refuses a global range whose size, or any one dimension of it, exceeds INT_MAX (the default
    // -fsycl-id-queries-range=int), and dpct::dim3 multiplies in unsigned: 16777217 rows times 256 wrapped to 256
    // work-items and one row was written. Launch at most max_groups0 groups along dimension 2; the kernel strides over
    // the rest.
    const int64_t        max_groups0 = std::max<int64_t>(1, (int64_t) INT_MAX / (bx * ne02 * ne03));
    const int64_t        launched0   = std::min(n_groups0, max_groups0);
    const sycl::range<3> local_range(1, 1, (size_t) bx);
    const sycl::range<3> global_range((size_t) ne03, (size_t) ne02, (size_t) (launched0 * bx));

    stream->submit([&](sycl::handler &cgh) {
        auto src0_data_ct0 = src0.resolve_ptr();
        auto dst_data_ct1  = dst.resolve_ptr();
        auto src0_nb_ct7   = src0.nb(0);
        auto src0_nb_ct8   = src0.nb(1);
        auto src0_nb_ct9   = src0.nb(2);
        auto src0_nb_ct10  = src0.nb(3);
        auto dst_nb_ct11   = dst.nb(0);
        auto dst_nb_ct12   = dst.nb(1);
        auto dst_nb_ct13   = dst.nb(2);
        auto dst_nb_ct14   = dst.nb(3);

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range), [=](sycl::nd_item<3> item_ct1) {
            pad_reflect_1d_kernel_f32(src0_data_ct0, dst_data_ct1, ne0, ne00, ne01_packed, ne02, ne03, src0_nb_ct7,
                                      src0_nb_ct8, src0_nb_ct9, src0_nb_ct10, dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                      dst_nb_ct14, p0, p1, n_groups0, item_ct1);
        });
    });

    GGML_UNUSED(dst_raw);
}
