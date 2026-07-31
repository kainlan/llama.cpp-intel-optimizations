// Host-only gate tests for the oneDNN SDPA layout planner. No GPU, no
// allocation, no oneDNN runtime -- it calls the planner directly.
//
// This file has been lost once, and its coverage failed once. Both are worth
// knowing before adding to it (llama.cpp-4jlv):
//
//  1. It WAS registered, in 3c8f296fd (tests/CMakeLists.txt:990-1002), and it
//     built and passed -- artifacts/kkxtv7-5/ctest-materialization.log records
//     "Test #100: test-sycl-fattn-onednn-gates ... Passed 0.11 sec". The
//     registration is present at 3c8f296fd and absent at HEAD, and nothing
//     announced its removal; the passing log was left behind as a fossil.
//
//     What is NOT established is HOW it was lost, and an earlier version of
//     this comment overstated it. Naming a culprit commit by "first commit in
//     3c8f296fd..HEAD whose tests/CMakeLists.txt lacks the string" does not
//     work: that traversal is topological, so on an integrated upstream line
//     it returns the first upstream commit to TOUCH the file, not one that
//     removed anything. It yields d3dce4e0a -- but d3dce4e0a's parent
//     4974bf53c does not have the registration either, so d3dce4e0a plainly
//     did not drop it, and it is a single-parent commit rather than the merge
//     the earlier text claimed. The loss is a consequence of integrating an
//     upstream line whose copy of this file never carried the fork's
//     registration; rebase vs merge-resolution vs manual overwrite is not
//     distinguished, and does not change the remedy.
//
//     (Recorded because the wrong version was a clean answer about the wrong
//     thing -- the same defect class this file's own gate exists to catch.)
//
//  2. Registration alone would NOT have caught the phi2 abort. Every case here
//     inherited `params.scale = 1.0f / 11.313708f` from mistral_like_params --
//     that is 1/sqrt(128), a CONFORMING scale -- and no case varied it. So all
//     11 cases passed on 2026-05-15 while the phi2 dispatch bug had already
//     been present since be45709a9 (2026-04-21). The planner had a test, the
//     test ran, and the scale axis simply was not in it.
//
// The lesson for anyone extending this file: a planner gate is only covered if
// some case makes the guarded quantity WRONG. A suite of cases that all supply
// well-formed inputs proves the accept path and nothing else.

#include "ggml-sycl/fattn.hpp"

#include <cmath>
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

// phi2 pre-scales Q by 1/sqrt(n_embd_head) inside the graph and then hands
// build_attn a kq_scale of 1.0 (src/models/phi2.cpp), so params.scale is 1.0
// rather than 1/sqrt(D). D=80 is phi2's head dim (n_embd 2560 / n_head 32).
static fattn_params phi2_like_params() {
    fattn_params params = mha_like_params();
    params.ne00         = 80;
    params.ne10         = 80;
    params.nb01         = params.ne00 * (int) sizeof(sycl::half);
    params.nb02         = params.nb01 * params.ne01;
    params.nb03         = params.nb02 * params.ne02;
    params.nb11         = params.ne10 * (int) sizeof(sycl::half);
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb21         = params.ne10 * (int) sizeof(sycl::half);
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    params.scale        = 1.0f;  // NOT 1/sqrt(80) — the whole point of the arch
    return params;
}

// The oneDNN partition bakes sqrt(D) in as the softmax divisor, so a model whose
// scale is not 1/sqrt(D) must never reach it. The execute path asserts this; the
// planner has to reject first or that assert aborts the process.
static bool test_planner_rejects_scale_not_inv_sqrt_d() {
    fattn_params params = phi2_like_params();
    const auto   plan   = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "params.scale != 1/sqrt(D) (phi2) must reject oneDNN, not abort at execute time");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                "scale reject reason should be explicit");
    return true;
}

// Same shape family, but GQA with a non-dense K/V stride — the plan kind that
// would otherwise be MATERIALIZE_REQUIRED. The scale gate must win over it,
// because the execute-time assert sits AFTER materialization.
static bool test_planner_rejects_scale_before_materialization() {
    fattn_params params = phi2_like_params();
    params.ne12         = 8;                                           // GQA
    params.nb11         = 4 * params.ne10 * (int) sizeof(sycl::half);  // nc_stride != D
    params.nb12         = params.nb11 * params.ne11;
    params.nb13         = (int64_t) params.nb12 * params.ne12;
    params.nb22         = params.nb21 * params.ne11;
    params.nb23         = (int64_t) params.nb22 * params.ne12;
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::REJECT,
                "a bad scale must reject even on the materialization path");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::SCALE_UNSUPPORTED,
                "scale gate must win over KV_NC_STRIDE_MISMATCH");
    return true;
}

// Guard the other direction: the scale gate must not reject a legitimate
// 1/sqrt(D) at a D the existing cases do not cover, or it silently costs
// throughput on every such model.
static bool test_planner_accepts_inv_sqrt_d_at_odd_d() {
    fattn_params params = phi2_like_params();
    params.scale        = 1.0f / std::sqrt(80.0f);
    const auto plan     = ggml_sycl_flash_attn_ext_onednn_plan(params, params.ne02, params.ne12, params.kv_is_fp8,
                                                               /*multi_seq=*/false);

    TEST_ASSERT(plan.kind == ggml_sycl_onednn_fa_layout_kind::DIRECT,
                "a legitimate 1/sqrt(D) scale at D=80 must still reach oneDNN");
    TEST_ASSERT(plan.reason == ggml_sycl_onednn_fa_layout_reason::OK, "1/sqrt(D) at D=80 should have OK reason");
    return true;
}

// The materialization descriptor is a second entry point into the planner; a
// rejected plan must not hand back a descriptor that invites the caller in.
static bool test_materialization_descriptor_rejects_bad_scale() {
    fattn_params                             params = phi2_like_params();
    ggml_sycl_onednn_fa_materialization_desc desc{};
    const bool ok = ggml_sycl_flash_attn_ext_onednn_materialization_desc(params, params.ne02, params.ne12,
                                                                         /*target_device=*/0, &desc);

    TEST_ASSERT(!ok, "materializer must refuse a shape the planner rejected on scale");
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
    ok &= test_planner_rejects_scale_not_inv_sqrt_d();
    ok &= test_planner_rejects_scale_before_materialization();
    ok &= test_planner_accepts_inv_sqrt_d_at_odd_d();
    ok &= test_materialization_descriptor_rejects_bad_scale();
    std::printf("SYCL fattn oneDNN gate tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
