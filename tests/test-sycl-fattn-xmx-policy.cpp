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

static fattn_params decode_params(int h_q, int h_kv) {
    fattn_params params{};
    params.Q_type    = GGML_TYPE_F16;
    params.K_type    = GGML_TYPE_F16;
    params.V_type    = GGML_TYPE_F16;
    params.mask_type = GGML_TYPE_F16;
    params.prec      = GGML_PREC_DEFAULT;
    params.ne00      = 128;
    params.ne01      = 1;
    params.ne02      = h_q;
    params.ne03      = 1;
    params.ne10      = 128;
    params.ne11      = 512;
    params.ne12      = h_kv;
    params.ne13      = 1;
    return params;
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

    std::printf("SYCL fattn policy tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
