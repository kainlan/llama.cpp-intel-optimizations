#include "ggml-sycl/fattn.hpp"

#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_SYCL) || !GGML_SYCL_DNNL
int main() {
    std::printf("GGML SYCL oneDNN not enabled; skipping test.\n");
    return 0;
}
#else

#    define TEST_ASSERT(cond, msg)                       \
        do {                                             \
            if (!(cond)) {                               \
                std::fprintf(stderr, "FAIL: %s\n", msg); \
                return false;                            \
            }                                            \
        } while (0)

static fattn_params mistral_like_params(int k_nc_stride_elems) {
    fattn_params params{};
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.scale     = 1.0f / 11.313708f;
    params.ne00      = 128;  // D
    params.ne01      = 512;  // prompt tokens
    params.ne02      = 32;   // Q heads
    params.ne03      = 1;
    params.ne10      = 128;
    params.ne11      = 512;  // KV tokens
    params.ne12      = 8;    // KV heads, so GQA
    params.ne13      = 1;
    params.nb01      = params.ne00 * (int) sizeof(sycl::half);
    params.nb02      = params.nb01 * params.ne01;
    params.nb03      = params.nb02 * params.ne02;
    params.nb11      = k_nc_stride_elems * (int) sizeof(sycl::half);
    params.nb12      = params.nb11 * params.ne11;
    params.nb13      = (int64_t) params.nb12 * params.ne12;
    params.nb21      = params.ne10 * (int) sizeof(sycl::half);
    params.nb22      = params.nb21 * params.ne11;
    params.nb23      = (int64_t) params.nb22 * params.ne12;
    params.prec      = GGML_PREC_F32;
    return params;
}

static fattn_params mha_like_params() {
    fattn_params params = mistral_like_params(/*k_nc_stride_elems=*/128);
    params.ne02         = 32;
    params.ne12         = 32;
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    return params;
}

static fattn_params mqa_like_params(int k_nc_stride_elems) {
    fattn_params params = mistral_like_params(k_nc_stride_elems);
    params.ne02         = 32;
    params.ne12         = 1;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    return params;
}

static bool test_gqa_nc_stride_mismatch_is_not_direct_onednn_eligible() {
    setenv("GGML_SYCL_FA_ONEDNN_ALLOW", "1", 1);

    fattn_params params   = mistral_like_params(/*k_nc_stride_elems=*/512);
    const bool   eligible = ggml_sycl_flash_attn_ext_onednn_eligible(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                     /*multi_seq=*/false);

    unsetenv("GGML_SYCL_FA_ONEDNN_ALLOW");
    TEST_ASSERT(!eligible, "nc_stride != D GQA must not be direct-eligible even with GGML_SYCL_FA_ONEDNN_ALLOW=1");
    return true;
}

static bool test_gqa_nc_stride_equal_d_remains_onednn_eligible() {
    unsetenv("GGML_SYCL_FA_ONEDNN_ALLOW");

    fattn_params params   = mistral_like_params(/*k_nc_stride_elems=*/128);
    const bool   eligible = ggml_sycl_flash_attn_ext_onednn_eligible(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                     /*multi_seq=*/false);

    TEST_ASSERT(eligible, "GQA nc_stride == D shape should remain oneDNN eligible");
    return true;
}

static bool test_planner_direct_mha_contiguous() {
    fattn_params params = mha_like_params();
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT, "contiguous MHA should use direct oneDNN layout");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "contiguous MHA should have OK reason");
    return true;
}

static bool test_planner_gqa_mismatch_requires_materialization() {
    fattn_params params = mistral_like_params(/*k_nc_stride_elems=*/512);
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED,
                "GQA nc_stride != D should require materialization before oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH,
                "GQA materialization reason should identify K/V nc stride mismatch");
    return true;
}

static bool test_planner_mqa_mismatch_requires_materialization() {
    fattn_params params = mqa_like_params(/*k_nc_stride_elems=*/512);
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::MATERIALIZE_REQUIRED,
                "MQA nc_stride != D should require materialization before oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::KV_NC_STRIDE_MISMATCH,
                "MQA materialization reason should identify K/V nc stride mismatch");
    return true;
}

static bool test_planner_rejects_unsupported_d() {
    fattn_params params = mha_like_params();
    params.ne00         = 1024;
    params.ne10         = 1024;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT, "D > 512 should reject oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::UNSUPPORTED_D,
                "unsupported D reject reason should be explicit");
    return true;
}

static bool test_planner_rejects_unproven_batch() {
    fattn_params params = mha_like_params();
    params.ne03         = 2;
    params.ne13         = 2;
    params.nb03         = params.nb02 * params.ne02;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "batch>1 tensors must reject oneDNN until batch descriptors/materialization are proven");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::BATCH_UNSUPPORTED,
                "batch>1 reject reason should be explicit");
    return true;
}

static bool test_planner_rejects_paged_layout() {
    fattn_params params     = mistral_like_params(/*k_nc_stride_elems=*/128);
    params.use_paged_attn   = true;
    params.use_paged_layout = true;
    const auto plan         = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                                   /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "paged K/V layouts must reject oneDNN until descriptor support is proven");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::PAGED_UNSUPPORTED,
                "paged-layout reject reason should be explicit");
    return true;
}

static bool test_materialization_descriptor_for_gqa_mismatch() {
    fattn_params                             params = mistral_like_params(/*k_nc_stride_elems=*/512);
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(ok, "GQA nc_stride != D should produce a materialization descriptor");
    TEST_ASSERT(desc.required, "GQA descriptor should mark materialization required");
    TEST_ASSERT(desc.target_device == 0, "descriptor should preserve target device");
    TEST_ASSERT(desc.D == params.ne00, "descriptor should preserve head dimension");
    TEST_ASSERT(desc.n_kv == params.ne11, "descriptor should preserve KV length");
    TEST_ASSERT(desc.H_kv == params.ne12, "descriptor should preserve KV heads");
    TEST_ASSERT(desc.bytes_per_tensor == (size_t) params.ne12 * params.ne11 * params.ne00 * sizeof(sycl::half),
                "descriptor should size one dense f16 K/V tensor");
    TEST_ASSERT(desc.k_target_nb1 == (int64_t) params.ne00 * (int64_t) sizeof(sycl::half),
                "materialized K token stride must be dense D");
    TEST_ASSERT(desc.v_target_nb1 == (int64_t) params.ne00 * (int64_t) sizeof(sycl::half),
                "materialized V token stride must be dense D");
    return true;
}

static bool test_materialization_descriptor_direct_mha_noop() {
    fattn_params                             params = mha_like_params();
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(ok, "direct MHA should still produce a descriptor");
    TEST_ASSERT(!desc.required, "direct MHA descriptor should be a no-op");
    TEST_ASSERT(desc.bytes_per_tensor == 0, "direct MHA no-op should not request materialization bytes");
    return true;
}

static bool test_materialization_descriptor_rejects_unsupported_layout() {
    fattn_params params = mqa_like_params(/*k_nc_stride_elems=*/512);
    params.ne02         = 30;
    params.ne12         = 8;
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(!ok, "materializer should reject non-integral GQA/MQA head ratios");
    return true;
}

int main() {
    bool ok = true;
    ok &= test_gqa_nc_stride_mismatch_is_not_direct_onednn_eligible();
    ok &= test_gqa_nc_stride_equal_d_remains_onednn_eligible();
    ok &= test_planner_direct_mha_contiguous();
    ok &= test_planner_gqa_mismatch_requires_materialization();
    ok &= test_planner_mqa_mismatch_requires_materialization();
    ok &= test_materialization_descriptor_for_gqa_mismatch();
    ok &= test_materialization_descriptor_direct_mha_noop();
    ok &= test_materialization_descriptor_rejects_unsupported_layout();
    ok &= test_planner_rejects_unsupported_d();
    ok &= test_planner_rejects_unproven_batch();
    ok &= test_planner_rejects_paged_layout();
    std::printf("SYCL fattn oneDNN gate tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
