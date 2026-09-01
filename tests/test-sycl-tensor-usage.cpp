// SYCL tensor usage inference test.
// Validates ggml_sycl_get_tensor_usage classification by tensor name.

#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif

#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool check_usage(ggml_tensor * tensor, tensor_usage expected, const char * label) {
    const tensor_usage got = ggml_sycl_get_tensor_usage(tensor);
    if (got != expected) {
        fprintf(stderr, "FAIL: %s expected=%d got=%d\n", label, (int) expected, (int) got);
        return false;
    }
    return true;
}

static bool check_dispatch_available(bool got, bool expected, const char * label) {
    if (got != expected) {
        fprintf(stderr, "FAIL: %s expected=%d got=%d\n", label, expected ? 1 : 0, got ? 1 : 0);
        return false;
    }
    return true;
}

int main() {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: CPU backend unavailable\n");
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_tensor * t_attn = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_attn, "blk.0.attn_q.weight");
    ggml_tensor * t_ffn = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_ffn, "blk.0.ffn_up.weight");
    ggml_tensor * t_moe_exp = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_moe_exp, "blk.0.ffn_gate_exps.weight");
    ggml_tensor * t_moe_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_moe_gate, "blk.0.ffn_gate_inp.weight");
    ggml_tensor * t_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_embed, "token_embd.weight");
    ggml_tensor * t_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_norm, "blk.0.attn_norm.weight");
    ggml_tensor * t_unknown = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 32);
    ggml_set_name(t_unknown, "mystery.weight");

    // gemma3n/gemma4 per-layer-embedding projection (llama.cpp-kmeq): this
    // weight has no "blk." prefix (it is shared across all layers, not
    // per-layer) and its name matches none of the attn/ffn/moe/embedding/norm
    // patterns above, so before the fix it fell through to UNKNOWN -> FFN
    // placement priority, letting the SYCL placement planner's greedy
    // VRAM-fill compete it against the model's bulk FFN/MoE weights despite
    // being tiny -- risking eviction to host and forcing
    // project_per_layer_inputs() onto the CPU every token. It must classify
    // as NORM (the "tiny, always keep on device" tier).
    ggml_tensor * t_ple_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 32);
    ggml_set_name(t_ple_proj, "per_layer_model_proj.weight");

    bool ok = true;
    ok = ok && check_usage(t_attn, tensor_usage::ATTENTION_WEIGHT, "attn");
    ok = ok && check_usage(t_ffn, tensor_usage::FFN_WEIGHT, "ffn");
    ok = ok && check_usage(t_moe_exp, tensor_usage::MOE_EXPERT_WEIGHT, "moe_expert");
    ok = ok && check_usage(t_moe_gate, tensor_usage::MOE_GATE, "moe_gate");
    ok = ok && check_usage(t_embed, tensor_usage::EMBEDDING, "embedding");
    ok = ok && check_usage(t_norm, tensor_usage::NORM, "norm");
    ok = ok && check_usage(t_unknown, tensor_usage::UNKNOWN, "unknown");
    ok = ok && check_usage(t_ple_proj, tensor_usage::NORM, "gemma_per_layer_model_proj");

    // llama.cpp-kmeq: ggml_sycl_bf16_weight_dispatch_available() coverage.
    // This predicate gates whether ggml_backend_sycl_device_supports_op()
    // admits a BF16 MUL_MAT for the ggml_sycl_bf16_weight_materialize_f32()
    // route (ggml-sycl.cpp). It is declared inline in common.hpp (not
    // `static` in ggml-sycl.cpp) specifically so this test calls the exact
    // same function production dispatch does, mirroring the fattn.hpp
    // pattern -- one definition, both callers -- rather than re-implementing
    // its logic here and risking drift.
    //
    // It needs a REAL is_weight()-true tensor (buffer usage == WEIGHTS) and
    // non-null data, which the no_alloc=true `ctx` above never allocates, so
    // this uses a second, separately-allocated context. The CPU buffer type
    // is deliberate: ggml_sycl_tensor_is_weight() and ggml_is_contiguous()
    // are backend-agnostic tensor-struct predicates (buffer-usage flag,
    // ne[]/nb[] shape), so this exercises the exact production logic
    // without needing a real SYCL device.
    {
        ggml_init_params params2 = {
            /*.mem_size   =*/4 * 1024 * 1024,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ggml_context * ctx2 = ggml_init(params2);
        if (!ctx2) {
            fprintf(stderr, "FAIL: ggml_init (dispatch-available ctx) failed\n");
            ok = false;
        } else {
            ggml_tensor * t_bf16 = ggml_new_tensor_2d(ctx2, GGML_TYPE_BF16, 8, 8);
            ggml_set_name(t_bf16, "test.dispatch.bf16.weight");
            ggml_tensor * t_f32 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, 8, 8);
            ggml_set_name(t_f32, "test.dispatch.f32.weight");
            ggml_tensor * t_bf16_noncontig = ggml_new_tensor_2d(ctx2, GGML_TYPE_BF16, 8, 8);
            ggml_set_name(t_bf16_noncontig, "test.dispatch.bf16.noncontig.weight");

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, ggml_backend_cpu_buffer_type());
            if (!buf) {
                fprintf(stderr, "FAIL: ggml_backend_alloc_ctx_tensors_from_buft failed\n");
                ok = false;
            } else {
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                // Break contiguity by widening nb[1] past the packed size --
                // the same shape a real permuted/viewed weight would have,
                // without needing graph machinery to produce one.
                t_bf16_noncontig->nb[1] = t_bf16_noncontig->nb[0] * t_bf16_noncontig->ne[0] + t_bf16_noncontig->nb[0];

                ok = ok && check_dispatch_available(ggml_sycl_bf16_weight_dispatch_available(t_bf16, /*device=*/0),
                                                    true, "bf16 contiguous weight -> available");
                ok = ok && check_dispatch_available(ggml_sycl_bf16_weight_dispatch_available(t_f32, /*device=*/0),
                                                    false, "non-bf16 weight -> unavailable");
                ok = ok &&
                     check_dispatch_available(ggml_sycl_bf16_weight_dispatch_available(t_bf16_noncontig, /*device=*/0),
                                              false, "non-contiguous bf16 weight -> unavailable");

                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx2);
        }
    }

    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("\nTensor usage test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
