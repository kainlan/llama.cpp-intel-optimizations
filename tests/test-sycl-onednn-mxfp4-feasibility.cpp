#if GGML_SYCL_DNNL
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_sycl.hpp>
#    include <sycl/sycl.hpp>
#endif

#include <cstdio>
#include <exception>

#if GGML_SYCL_DNNL
static dnnl::engine make_probe_engine() {
    sycl::device  dev{ sycl::default_selector_v };
    sycl::context ctx{ dev };
    return dnnl::sycl_interop::make_engine(dev, ctx);
}

static void probe_f4_e2m1_matmul(dnnl::engine & eng) {
    constexpr int64_t m = 1;
    constexpr int64_t k = 2880;
    constexpr int64_t n = 2880;

    const dnnl::memory::desc a_md({ m, k }, dnnl::memory::data_type::f16, { k, 1 });
    const dnnl::memory::desc b_md({ k, n }, dnnl::memory::data_type::f4_e2m1, dnnl::memory::format_tag::any);
    const dnnl::memory::desc c_md({ m, n }, dnnl::memory::data_type::f16, { n, 1 });

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    try {
        const auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        std::printf("onednn.mxfp4.f4_e2m1.matmul supported m=%lld n=%lld k=%lld weights_bytes=%zu scratchpad_bytes=%zu\n",
                    (long long) m, (long long) n, (long long) k, pd.weights_desc().get_size(),
                    pd.scratchpad_desc().get_size());
    } catch (const dnnl::error & e) {
        std::printf("onednn.mxfp4.f4_e2m1.matmul unsupported reason=%s\n", e.what());
    } catch (const std::exception & e) {
        std::printf("onednn.mxfp4.f4_e2m1.matmul unsupported reason=%s\n", e.what());
    }
}

static void probe_e8m0_scale_descriptor() {
    constexpr int64_t k              = 2880;
    constexpr int64_t n              = 2880;
    constexpr int64_t mx_block       = 32;
    constexpr int64_t scale_k_blocks = k / mx_block;

    try {
        const dnnl::memory::desc scale_md({ scale_k_blocks, n }, dnnl::memory::data_type::e8m0, { n, 1 });
        std::printf("onednn.mxfp4.e8m0.scale_descriptor supported k=%lld n=%lld scale_k_blocks=%lld bytes=%zu\n",
                    (long long) k, (long long) n, (long long) scale_k_blocks, scale_md.get_size());
    } catch (const dnnl::error & e) {
        std::printf("onednn.mxfp4.e8m0.scale_descriptor unsupported reason=%s\n", e.what());
    } catch (const std::exception & e) {
        std::printf("onednn.mxfp4.e8m0.scale_descriptor unsupported reason=%s\n", e.what());
    }
}

static void probe_s4_woq_matmul(dnnl::engine & eng) {
    constexpr int64_t m          = 1;
    constexpr int64_t k          = 2880;
    constexpr int64_t n          = 2880;
    constexpr int64_t group_size = 32;

    const dnnl::memory::desc a_md({ m, k }, dnnl::memory::data_type::f16, { k, 1 });
    const dnnl::memory::desc b_md({ k, n }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::any);
    const dnnl::memory::desc c_md({ m, n }, dnnl::memory::data_type::f16, { n, 1 });

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const int   mask       = (1 << 0) | (1 << 1);
    dnnl_dims_t group_dims = { group_size, 1 };
    if (dnnl_primitive_attr_set_scales(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                       dnnl::memory::convert_to_c(dnnl::memory::data_type::f32)) != dnnl_success) {
        std::puts("onednn.mxfp4.s4_woq.matmul unsupported reason=set_scales_failed");
        return;
    }
    if (dnnl_primitive_attr_set_zero_points(attr.get(), DNNL_ARG_WEIGHTS, mask, 2, group_dims,
                                            dnnl::memory::convert_to_c(dnnl::memory::data_type::s8)) != dnnl_success) {
        std::puts("onednn.mxfp4.s4_woq.matmul unsupported reason=set_zero_points_failed");
        return;
    }

    try {
        const auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        std::printf("onednn.mxfp4.s4_woq.matmul supported m=%lld n=%lld k=%lld group=%lld weights_bytes=%zu scratchpad_bytes=%zu\n",
                    (long long) m, (long long) n, (long long) k, (long long) group_size,
                    pd.weights_desc().get_size(), pd.scratchpad_desc().get_size());
    } catch (const dnnl::error & e) {
        std::printf("onednn.mxfp4.s4_woq.matmul unsupported reason=%s\n", e.what());
    } catch (const std::exception & e) {
        std::printf("onednn.mxfp4.s4_woq.matmul unsupported reason=%s\n", e.what());
    }
}
#endif

int main() {
#if GGML_SYCL_DNNL
    probe_e8m0_scale_descriptor();

    try {
        dnnl::engine eng = make_probe_engine();
        probe_f4_e2m1_matmul(eng);
        probe_s4_woq_matmul(eng);
    } catch (const dnnl::error & e) {
        std::printf("onednn.mxfp4.probe setup_failed reason=%s\n", e.what());
        std::puts("onednn.mxfp4.f4_e2m1.matmul unsupported reason=setup_failed");
        std::puts("onednn.mxfp4.s4_woq.matmul unsupported reason=setup_failed");
    } catch (const std::exception & e) {
        std::printf("onednn.mxfp4.probe setup_failed reason=%s\n", e.what());
        std::puts("onednn.mxfp4.f4_e2m1.matmul unsupported reason=setup_failed");
        std::puts("onednn.mxfp4.s4_woq.matmul unsupported reason=setup_failed");
    }
#else
    std::puts("onednn.mxfp4.probe skipped GGML_SYCL_DNNL=0");
#endif
    return 0;
}
