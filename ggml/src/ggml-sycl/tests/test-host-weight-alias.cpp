//
// Test: host-weight registry name-collision classifier (llama.cpp-ttws).
//
// ggml_backend_sycl_register_host_weight_tensor keys its registry on
// owner + tensor NAME. llama-model-loader's TENSOR_DUPLICATED path creates a
// SECOND ggml_tensor object under the same name whenever a model ties its
// output head to token_embd (Gemma 3n / Gemma 4 do; src/models/gemma4.cpp:44-50
// even creates the alias before the original, so the loader's reuse lookup can
// never hit). Both objects describe the same GGUF bytes, and the registry's
// reconcile already merges them onto one extra -- the only defect was that it
// WARNED about the expected case on every Gemma 4 load.
//
// classify_host_weight_alias() is the pure decision the registry consults
// before warning. Host-only: it reads ggml_tensor metadata only, so no GPU, no
// device, no libggml-sycl link -- just ggml-base for ggml_init/ggml_nbytes.
// Every failing row is reported (not just the first) so a RED run lists the
// whole defect at once.
//
// Written RED before the header existed; the RED text is recorded in this
// ticket's commit body.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "host-weight-alias.hpp"

#include <cstdio>
#include <cstring>

using ggml_sycl::detail::classify_host_weight_alias;
using ggml_sycl::detail::host_weight_alias_kind;
using ggml_sycl::detail::host_weight_alias_kind_name;

static int g_failures = 0;

static void expect(const char * label, host_weight_alias_kind got, host_weight_alias_kind want) {
    if (got != want) {
        printf("  FAIL: %s: got %s, want %s\n", label, host_weight_alias_kind_name(got),
               host_weight_alias_kind_name(want));
        g_failures++;
    } else {
        printf("  ok:   %s -> %s\n", label, host_weight_alias_kind_name(got));
    }
}

int main() {
    // Two no_alloc contexts stand in for the loader's two per-buffer-type
    // contexts (OUTPUT-layer device ctx and INPUT-layer host ctx).
    struct ggml_init_params params = {
        /*.mem_size   =*/16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx_out = ggml_init(params);
    ggml_context * ctx_in  = ggml_init(params);
    if (!ctx_out || !ctx_in) {
        printf("FAIL: ggml_init\n");
        return 1;
    }

    // Gemma 4 E4B token_embd.weight: Q8_0, {n_embd=2560, n_vocab=262144}.
    const int64_t n_embd  = 2560;
    const int64_t n_vocab = 262144;

    ggml_tensor * output_alias = ggml_new_tensor_2d(ctx_out, GGML_TYPE_Q8_0, n_embd, n_vocab);
    ggml_set_name(output_alias, "token_embd.weight");
    ggml_tensor * tok_embd = ggml_new_tensor_2d(ctx_in, GGML_TYPE_Q8_0, n_embd, n_vocab);
    ggml_set_name(tok_embd, "token_embd.weight");

    // 1. the Gemma 4 case: two objects, one GGUF weight -> alias, not a mismatch
    expect("gemma4 tied output alias vs tok_embd", classify_host_weight_alias(output_alias, tok_embd),
           host_weight_alias_kind::ALIAS_SAME_BYTES);
    expect("symmetric", classify_host_weight_alias(tok_embd, output_alias), host_weight_alias_kind::ALIAS_SAME_BYTES);

    // 2. the same object registered twice (loader reuse path) is SAME_TENSOR
    expect("same object", classify_host_weight_alias(tok_embd, tok_embd), host_weight_alias_kind::SAME_TENSOR);

    // 3. same name, different shape: a genuine conflict
    ggml_tensor * reshaped = ggml_new_tensor_2d(ctx_in, GGML_TYPE_Q8_0, n_embd * 2, n_vocab / 2);
    ggml_set_name(reshaped, "token_embd.weight");
    expect("same name, different ne[] (same nbytes)", classify_host_weight_alias(tok_embd, reshaped),
           host_weight_alias_kind::DIVERGENT);

    // 4. same name and shape, different type
    ggml_tensor * retyped = ggml_new_tensor_2d(ctx_in, GGML_TYPE_Q4_0, n_embd, n_vocab);
    ggml_set_name(retyped, "token_embd.weight");
    expect("same name and ne[], different type", classify_host_weight_alias(tok_embd, retyped),
           host_weight_alias_kind::DIVERGENT);

    // 5. different name (the registry key would differ; the classifier must not
    //    call two different weights an alias regardless)
    ggml_tensor * other = ggml_new_tensor_2d(ctx_in, GGML_TYPE_Q8_0, n_embd, n_vocab);
    ggml_set_name(other, "per_layer_token_embd.weight");
    expect("different name", classify_host_weight_alias(tok_embd, other), host_weight_alias_kind::DIVERGENT);

    // 6. a 3-D expert tensor vs a 2-D one of equal byte count
    ggml_tensor * moe = ggml_new_tensor_3d(ctx_in, GGML_TYPE_Q8_0, n_embd, n_vocab / 4, 4);
    ggml_set_name(moe, "token_embd.weight");
    expect("2-D vs 3-D, equal nbytes", classify_host_weight_alias(tok_embd, moe), host_weight_alias_kind::DIVERGENT);

    // 7. null on either side is never an alias
    expect("null existing", classify_host_weight_alias(nullptr, tok_embd), host_weight_alias_kind::DIVERGENT);
    expect("null incoming", classify_host_weight_alias(tok_embd, nullptr), host_weight_alias_kind::DIVERGENT);

    // 8. the per-layer rope_freqs duplicate (the other TENSOR_DUPLICATED user)
    ggml_tensor * rope_a = ggml_new_tensor_1d(ctx_out, GGML_TYPE_F32, 64);
    ggml_set_name(rope_a, "rope_freqs.weight");
    ggml_tensor * rope_b = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, 64);
    ggml_set_name(rope_b, "rope_freqs.weight");
    expect("rope_freqs duplicate", classify_host_weight_alias(rope_a, rope_b),
           host_weight_alias_kind::ALIAS_SAME_BYTES);

    // 9. every kind has a distinct printable name
    if (std::strcmp(host_weight_alias_kind_name(host_weight_alias_kind::SAME_TENSOR),
                    host_weight_alias_kind_name(host_weight_alias_kind::ALIAS_SAME_BYTES)) == 0 ||
        std::strcmp(host_weight_alias_kind_name(host_weight_alias_kind::ALIAS_SAME_BYTES),
                    host_weight_alias_kind_name(host_weight_alias_kind::DIVERGENT)) == 0) {
        printf("  FAIL: kind names are not distinct\n");
        g_failures++;
    }

    ggml_free(ctx_in);
    ggml_free(ctx_out);

    if (g_failures) {
        printf("FAIL: host-weight-alias: %d row(s)\n", g_failures);
        return 1;
    }
    printf("PASS: host-weight-alias: 10 rows + name distinctness\n");
    return 0;
}
