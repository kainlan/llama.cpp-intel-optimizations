#include "ggml-sycl/fattn-onednn.hpp"
#include "ggml-sycl/unified-cache.hpp"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::printf("GGML SYCL oneDNN not enabled; skipping test.\n");
    return 0;
}
#else

#    define TEST_ASSERT(cond, msg)                    \
        do {                                          \
            if (!(cond)) {                            \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                         \
            }                                         \
        } while (0)

static fattn_params tiny_params(int H_q, int H_kv, int D, int n_kv, int k_nc_stride, int v_nc_stride) {
    fattn_params params{};
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.scale     = 1.0f / sqrtf((float) D);
    params.ne00      = D;
    params.ne01      = 8;
    params.ne02      = H_q;
    params.ne03      = 1;
    params.ne10      = D;
    params.ne11      = n_kv;
    params.ne12      = H_kv;
    params.ne13      = 1;
    params.nb01      = D * (int) sizeof(sycl::half);
    params.nb02      = params.nb01 * params.ne01;
    params.nb03      = params.nb02 * H_q;
    params.nb11      = k_nc_stride * (int) sizeof(sycl::half);
    params.nb12      = params.nb11 * n_kv;
    params.nb13      = (int64_t) params.nb12 * H_kv;
    params.nb21      = v_nc_stride * (int) sizeof(sycl::half);
    params.nb22      = params.nb21 * n_kv;
    params.nb23      = (int64_t) params.nb22 * H_kv;
    params.prec      = GGML_PREC_F32;
    return params;
}

static bool test_gqa_byte_index_mapping() {
    fattn_params params = tiny_params(/*H_q=*/8, /*H_kv=*/2, /*D=*/4, /*n_kv=*/3,
                                      /*k_nc_stride=*/7, /*v_nc_stride=*/6);
    ggml_sycl_onednn_fa_materialization_desc desc{};
    TEST_ASSERT(ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12, 0, &desc),
                "GQA materialization descriptor failed");
    TEST_ASSERT(desc.required, "GQA mismatch should require materialization");

    const int h = 1;
    const int t = 2;
    const int d = 3;
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_src_offset(desc, false, h, t, d) ==
                    (size_t) h * params.nb12 + (size_t) t * params.nb11 + (size_t) d * sizeof(sycl::half),
                "GQA K source byte offset mismatch");
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_src_offset(desc, true, h, t, d) ==
                    (size_t) h * params.nb22 + (size_t) t * params.nb21 + (size_t) d * sizeof(sycl::half),
                "GQA V source byte offset mismatch");
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_dst_offset(desc, h, t, d) ==
                    ((size_t) h * params.ne11 * params.ne00 + (size_t) t * params.ne00 + (size_t) d) *
                        sizeof(sycl::half),
                "GQA target byte offset mismatch");
    return true;
}

static bool test_mqa_byte_index_mapping() {
    fattn_params params = tiny_params(/*H_q=*/8, /*H_kv=*/1, /*D=*/5, /*n_kv=*/4,
                                      /*k_nc_stride=*/9, /*v_nc_stride=*/8);
    ggml_sycl_onednn_fa_materialization_desc desc{};
    TEST_ASSERT(ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12, 0, &desc),
                "MQA materialization descriptor failed");
    TEST_ASSERT(desc.required, "MQA mismatch should require materialization");
    TEST_ASSERT(desc.bytes_per_tensor == (size_t) params.ne12 * params.ne11 * params.ne00 * sizeof(sycl::half),
                "MQA target byte size mismatch");

    const int h = 0;
    const int t = 3;
    const int d = 4;
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_src_offset(desc, false, h, t, d) ==
                    (size_t) t * params.nb11 + (size_t) d * sizeof(sycl::half),
                "MQA K source byte offset mismatch");
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_dst_offset(desc, h, t, d) ==
                    ((size_t) t * params.ne00 + (size_t) d) * sizeof(sycl::half),
                "MQA target byte offset mismatch");
    return true;
}

static bool test_non_monotonic_source_stride_descriptor() {
    fattn_params params = tiny_params(/*H_q=*/8, /*H_kv=*/2, /*D=*/4, /*n_kv=*/3,
                                      /*k_nc_stride=*/7, /*v_nc_stride=*/6);
    params.nb12 = params.ne00 * (int) sizeof(sycl::half);
    params.nb22 = params.ne00 * (int) sizeof(sycl::half);

    ggml_sycl_onednn_fa_materialization_desc desc{};
    TEST_ASSERT(ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12, 0, &desc),
                "materialization descriptor should accept strided ggml views with non-monotonic source strides");
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_src_offset(desc, false, /*h=*/1, /*t=*/2, /*d=*/3) ==
                    (size_t) params.nb12 + (size_t) 2 * params.nb11 + (size_t) 3 * sizeof(sycl::half),
                "non-monotonic K source byte offset mismatch");
    TEST_ASSERT(ggml_sycl_onednn_fa_materialized_src_offset(desc, true, /*h=*/1, /*t=*/2, /*d=*/3) ==
                    (size_t) params.nb22 + (size_t) 2 * params.nb21 + (size_t) 3 * sizeof(sycl::half),
                "non-monotonic V source byte offset mismatch");
    return true;
}

static bool test_materialize_f16_device_mapping_and_handle() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0", 1);
    }

    constexpr int H_kv = 2;
    constexpr int D = 4;
    constexpr int n_kv = 3;
    constexpr int k_stride = 7;
    constexpr int v_stride = 6;
    fattn_params params = tiny_params(/*H_q=*/8, H_kv, D, n_kv, k_stride, v_stride);
    ggml_sycl_onednn_fa_materialization_desc desc{};
    TEST_ASSERT(ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12, 0, &desc),
                "materialization descriptor failed");

    sycl::queue & q = ggml_sycl_get_device(0).default_queue();
    const size_t k_src_bytes = (size_t) H_kv * n_kv * k_stride * sizeof(sycl::half);
    const size_t v_src_bytes = (size_t) H_kv * n_kv * v_stride * sizeof(sycl::half);
    std::vector<uint16_t> k_host(k_src_bytes / sizeof(uint16_t), 0);
    std::vector<uint16_t> v_host(v_src_bytes / sizeof(uint16_t), 0);
    for (int h = 0; h < H_kv; ++h) {
        for (int t = 0; t < n_kv; ++t) {
            for (int d = 0; d < D; ++d) {
                k_host[(h * n_kv * k_stride) + (t * k_stride) + d] = (uint16_t) (1000 + h * 100 + t * 10 + d);
                v_host[(h * n_kv * v_stride) + (t * v_stride) + d] = (uint16_t) (2000 + h * 100 + t * 10 + d);
            }
        }
    }

    ggml_sycl::alloc_request k_req{};
    k_req.queue = &q;
    k_req.device = 0;
    k_req.size = k_src_bytes;
    k_req.intent.role = ggml_sycl::alloc_role::KV;
    k_req.intent.category = ggml_sycl::runtime_category::KV_CACHE;
    k_req.intent.constraints.must_device = true;
    ggml_sycl::alloc_request v_req = k_req;
    v_req.size = v_src_bytes;

    ggml_sycl::scoped_unified_alloc k_src(k_req);
    ggml_sycl::scoped_unified_alloc v_src(v_req);
    TEST_ASSERT(k_src.get() && v_src.get(), "source unified allocations failed");
    q.memcpy(k_src.get(), k_host.data(), k_src_bytes).wait();
    q.memcpy(v_src.get(), v_host.data(), v_src_bytes).wait();
    params.K = static_cast<const char *>(k_src.get());
    params.V = static_cast<const char *>(v_src.get());

    ggml_sycl_onednn_fa_materialized_kv materialized{};
    TEST_ASSERT(ggml_sycl_flash_attn_ext_onednn_materialize_kv(desc, params, q, &materialized),
                "device K/V materialization failed");
    auto k_resolved = materialized.K.resolve(0);
    auto v_resolved = materialized.V.resolve(0);
    TEST_ASSERT(k_resolved && v_resolved, "materialized handles must resolve on target device");
    TEST_ASSERT(k_resolved.on_device && v_resolved.on_device, "materialized handles must be device-resident");
    TEST_ASSERT(!materialized.K.resolve(1), "materialized handle must fail closed on wrong device");

    std::vector<uint16_t> k_out(desc.bytes_per_tensor / sizeof(uint16_t), 0);
    std::vector<uint16_t> v_out(desc.bytes_per_tensor / sizeof(uint16_t), 0);
    q.memcpy(k_out.data(), k_resolved.ptr, desc.bytes_per_tensor).wait();
    q.memcpy(v_out.data(), v_resolved.ptr, desc.bytes_per_tensor).wait();

    for (int h = 0; h < H_kv; ++h) {
        for (int t = 0; t < n_kv; ++t) {
            for (int d = 0; d < D; ++d) {
                const size_t dst_idx = (size_t) h * n_kv * D + (size_t) t * D + d;
                TEST_ASSERT(k_out[dst_idx] == (uint16_t) (1000 + h * 100 + t * 10 + d),
                            "materialized K value mismatch");
                TEST_ASSERT(v_out[dst_idx] == (uint16_t) (2000 + h * 100 + t * 10 + d),
                            "materialized V value mismatch");
            }
        }
    }

    return true;
}

int main() {
    bool ok = true;
    ok &= test_gqa_byte_index_mapping();
    ok &= test_mqa_byte_index_mapping();
    ok &= test_non_monotonic_source_stride_descriptor();
    ok &= test_materialize_f16_device_mapping_and_handle();
    std::printf("SYCL oneDNN FA materialization tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
