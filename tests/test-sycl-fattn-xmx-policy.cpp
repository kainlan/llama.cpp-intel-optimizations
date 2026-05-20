#include "ggml-sycl/fattn-xmx-f16.hpp"
#include "ggml-sycl/fattn.hpp"

#include <cstdio>

static bool expect_eq(int got, int want, const char * name) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s got %d want %d\n", name, got, want);
        return false;
    }
    return true;
}

static bool expect_eq_size(size_t got, size_t want, const char * name) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s got %zu want %zu\n", name, got, want);
        return false;
    }
    return true;
}

static bool expect_decode_policy(const fattn_params &                 params,
                                 int                                  D,
                                 bool                                 want_fast,
                                 ggml_sycl_fattn_decode_policy_reason want_reason,
                                 const char *                         name) {
    const ggml_sycl_fattn_decode_policy got = ggml_sycl_fattn_fast_decode_policy(params, D);
    if (got.fast_esimd_safe != want_fast || got.reason != want_reason) {
        std::fprintf(stderr, "FAIL: %s got fast=%d reason=%d want fast=%d reason=%d\n", name, (int) got.fast_esimd_safe,
                     (int) got.reason, (int) want_fast, (int) want_reason);
        return false;
    }
    return true;
}

static bool expect_xmx_decode_kv_plan(const fattn_params &                        params,
                                      int                                         D,
                                      const ggml_sycl_fattn_xmx_decode_kv_caps &  caps,
                                      ggml_sycl_fattn_xmx_decode_kv_layout_kind   want_kind,
                                      ggml_sycl_fattn_xmx_decode_kv_layout_reason want_reason,
                                      int                                         want_tk,
                                      const char *                                name) {
    const ggml_sycl_fattn_xmx_decode_kv_layout_plan got =
        ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(params, D, caps);
    if (got.kind != want_kind || got.reason != want_reason || got.preferred_tk != want_tk) {
        std::fprintf(stderr, "FAIL: %s got kind=%d reason=%d tk=%d want kind=%d reason=%d tk=%d\n", name,
                     (int) got.kind, (int) got.reason, got.preferred_tk, (int) want_kind, (int) want_reason, want_tk);
        return false;
    }
    return true;
}

static fattn_params decode_params_dim(int h_q, int h_kv, int D) {
    fattn_params params{};
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.prec      = GGML_PREC_DEFAULT;
    params.ne00      = D;
    params.ne01      = 1;
    params.ne02      = h_q;
    params.ne03      = 1;
    params.ne10      = D;
    params.ne11      = 512;
    params.ne12      = h_kv;
    params.ne13      = 1;
    params.nb11      = D * (int) sizeof(sycl::half);
    params.nb12      = params.ne11 * params.nb11;
    params.nb13      = (int64_t) params.ne12 * params.nb12;
    return params;
}

static fattn_params decode_params(int h_q, int h_kv) {
    return decode_params_dim(h_q, h_kv, 128);
}

int main() {
    bool ok = true;

    ok &= expect_eq(ggml_sycl_fattn_xmx_v1_select_batch_kv(/*D=*/128, /*ncols=*/8, /*local_mem_size=*/96 * 1024), 48,
                    "D128 ncols8 uses larger batch when local memory permits");
    ok &= expect_eq(ggml_sycl_fattn_xmx_v1_select_batch_kv(/*D=*/128, /*ncols=*/8, /*local_mem_size=*/48 * 1024), 32,
                    "D128 ncols8 falls back on smaller local memory");
    ok &= expect_eq(ggml_sycl_fattn_xmx_v1_select_batch_kv(/*D=*/64, /*ncols=*/8, /*local_mem_size=*/96 * 1024), 32,
                    "non-D128 shapes retain conservative batch");

    ok &=
        expect_decode_policy(decode_params(/*h_q=*/32, /*h_kv=*/8), 128, true, ggml_sycl_fattn_decode_policy_reason::OK,
                             "Mistral-like f16 GQA single-query decode may use fast ESIMD");
    ok &= expect_decode_policy(decode_params(/*h_q=*/32, /*h_kv=*/32), 128, true,
                               ggml_sycl_fattn_decode_policy_reason::OK,
                               "MHA f16 single-query decode may use fast ESIMD");

    {
        fattn_params params = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.ne01         = 2;
        ok &= expect_decode_policy(params, 128, false, ggml_sycl_fattn_decode_policy_reason::NOT_SINGLE_QUERY,
                                   "multi-query prompt/decode stays out of fast single-query policy");
    }
    {
        fattn_params params = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.sinks        = reinterpret_cast<const char *>(0x1);
        ok &= expect_decode_policy(params, 128, true, ggml_sycl_fattn_decode_policy_reason::OK,
                                   "attention sinks are implemented by fast ESIMD decode");
    }
    {
        fattn_params params  = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.logit_softcap = 1.0f;
        ok &= expect_decode_policy(params, 128, true, ggml_sycl_fattn_decode_policy_reason::OK,
                                   "softcap is implemented by fast ESIMD decode");
    }
    {
        fattn_params params = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.max_bias     = 8.0f;
        ok &= expect_decode_policy(params, 128, true, ggml_sycl_fattn_decode_policy_reason::OK,
                                   "ALiBi is implemented by fast ESIMD decode");
    }
    {
        fattn_params params = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.kv_is_fp8    = true;
        ok &= expect_decode_policy(params, 128, false, ggml_sycl_fattn_decode_policy_reason::FP8_KV_UNSUPPORTED,
                                   "FP8 KV decode stays conservative");
    }
    {
        fattn_params params   = decode_params(/*h_q=*/32, /*h_kv=*/8);
        params.use_paged_attn = true;
        ok &= expect_decode_policy(params, 128, false, ggml_sycl_fattn_decode_policy_reason::PAGED_UNSUPPORTED,
                                   "paged decode stays conservative");
    }
    {
        fattn_params params = decode_params(/*h_q=*/30, /*h_kv=*/8);
        ok &= expect_decode_policy(params, 128, false, ggml_sycl_fattn_decode_policy_reason::HEAD_RATIO_UNSUPPORTED,
                                   "non-integral GQA head ratio stays conservative");
    }
    {
        ggml_sycl_fattn_xmx_decode_kv_caps caps;
        caps.m1n64_k16_supported = true;
        caps.m1n64_k32_supported = true;
        caps.local_mem_size      = 128 * 1024;
        caps.k_device_resident   = true;
        caps.v_device_resident   = true;

        const fattn_params params = decode_params_dim(/*h_q=*/64, /*h_kv=*/8, /*D=*/64);
        ok &=
            expect_xmx_decode_kv_plan(params, 64, caps, ggml_sycl_fattn_xmx_decode_kv_layout_kind::PACKED_K_MEM_HANDLE,
                                      ggml_sycl_fattn_xmx_decode_kv_layout_reason::OK, 32,
                                      "GPT-OSS decode can plan a packed-K mem_handle layout from caps");
        const ggml_sycl_fattn_xmx_decode_kv_layout_plan plan =
            ggml_sycl_fattn_xmx_decode_kv_layout_plan_from_caps(params, 64, caps);
        ok &= expect_eq((int) plan.n_rep, 8, "GPT-OSS GQA ratio captured in XMX decode KV plan");
        ok &= expect_eq((int) plan.kv_block_tokens, 64, "XMX decode KV plan uses measured 64-token block");
        ok &= expect_eq((int) plan.source_k_bytes_per_block, 8192, "source K block byte count is explicit");
        ok &= expect_eq((int) plan.packed_k_bytes_per_block, 16384, "packed K block byte count is explicit");
        ok &= expect_eq((int) plan.packed_k_overhead_per_block, 8192, "packed K overhead is explicit");

        ggml_sycl_fattn_xmx_packed_k_materialization_desc desc{};
        const bool                                        desc_ok =
            ggml_sycl_fattn_xmx_packed_k_materialization_desc_from_plan(params, plan, /*target_device=*/0, &desc);
        ok &= expect_eq((int) desc_ok, 1, "packed K materialization descriptor accepts planned D64 f16 K");
        ok &= expect_eq(desc.n_blocks, 8, "packed K materializer computes 64-token block count");
        ok &= expect_eq_size(desc.total_blocks, 64, "packed K materializer counts batch*heads*blocks");
        ok &= expect_eq_size(desc.total_packed_bytes, 64 * 16384, "packed K materializer budgets total sidecar bytes");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_block_offset_bytes(desc, 0, 0, 1), 16384,
                             "packed K block stride is one packed block");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_block_offset_bytes(desc, 0, 1, 0), 8 * 16384,
                             "packed K head stride is n_blocks packed blocks");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_element_offset_half(0, 0), 0,
                             "packed K offset maps token 0 d0 to first half");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_element_offset_half(1, 0), 2,
                             "packed K offset maps compact lane 1 to packed column 1");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_element_offset_half(8, 0), 32,
                             "packed K offset skips inactive columns after active lane group");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_element_offset_half(31, 63), 4079,
                             "packed K offset covers last token in first compact half");
        ok &= expect_eq_size(ggml_sycl_fattn_xmx_packed_k_element_offset_half(32, 0), 4096,
                             "packed K offset starts second compact half at half-block boundary");
    }
    {
        ggml_sycl_fattn_xmx_decode_kv_caps caps;
        caps.m1n64_k16_supported = true;
        caps.local_mem_size      = 128 * 1024;
        caps.k_device_resident   = false;
        caps.v_device_resident   = true;
        ok &= expect_xmx_decode_kv_plan(decode_params_dim(/*h_q=*/64, /*h_kv=*/8, /*D=*/64), 64, caps,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_kind::REJECT,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_reason::KV_NOT_DEVICE_RESIDENT, 0,
                                        "host-resident K rejects packed device XMX decode layout");
    }
    {
        ggml_sycl_fattn_xmx_decode_kv_caps caps;
        caps.m1n64_k16_supported = true;
        caps.local_mem_size      = 4 * 1024;
        caps.k_device_resident   = true;
        caps.v_device_resident   = true;
        ok &= expect_xmx_decode_kv_plan(decode_params_dim(/*h_q=*/64, /*h_kv=*/8, /*D=*/64), 64, caps,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_kind::REJECT,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_reason::LOCAL_MEM_UNSUPPORTED, 0,
                                        "insufficient SLM rejects packed device XMX decode layout");
    }
    {
        ggml_sycl_fattn_xmx_decode_kv_caps caps;
        caps.m1n64_k16_supported = false;
        caps.local_mem_size      = 128 * 1024;
        caps.k_device_resident   = true;
        caps.v_device_resident   = true;
        ok &= expect_xmx_decode_kv_plan(decode_params_dim(/*h_q=*/64, /*h_kv=*/8, /*D=*/64), 64, caps,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_kind::REJECT,
                                        ggml_sycl_fattn_xmx_decode_kv_layout_reason::DEVICE_XMX_M1N64_UNSUPPORTED, 0,
                                        "missing M=1,N=64,K=16 matrix capability rejects XMX decode KV layout");
    }

    std::printf("SYCL fattn policy tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
