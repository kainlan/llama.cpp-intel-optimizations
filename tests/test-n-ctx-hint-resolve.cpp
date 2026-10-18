// llama.cpp-7fmt: the context a load holds KV room for, resolved from llama_model_params::n_ctx_hint. Pure host test,
// no model, no device.
//
// Without -c, common asks for the model's default context (LLAMA_N_CTX_HINT_DEFAULT): the context llama_context
// creates for n_ctx 0, which is n_ctx_train padded to 256 (src/llama-context.cpp). The load must hold the room for that
// context, and say it came from the default. 0 keeps its meaning (no request: no room), and an explicit hint is used
// as given, padded the same way. The sentinel is 0xFFFFFFFF, so read as a literal context it would pad (wrapping) to
// 0 and hold no room, which is the defect.

#include "../src/llama-n-ctx-hint.h"

#include <cstdio>

static int g_failures = 0;

static void expect_resolution(uint32_t     n_ctx_hint,
                              uint32_t     n_ctx_train,
                              uint32_t     want_n_ctx,
                              bool         want_default,
                              const char * what) {
    const llama_n_ctx_hint_resolution got = llama_n_ctx_hint_resolve(n_ctx_hint, n_ctx_train);
    if (got.n_ctx_context != want_n_ctx || got.is_default != want_default) {
        std::fprintf(stderr, "FAIL: %s: hint=%u n_ctx_train=%u -> n_ctx_context=%u is_default=%d, want %u %d\n", what,
                     n_ctx_hint, n_ctx_train, got.n_ctx_context, got.is_default ? 1 : 0, want_n_ctx,
                     want_default ? 1 : 0);
        g_failures++;
    }
}

int main() {
    // Qwen3.8-Flash-Next and GPT-OSS 20B training contexts, and one that is not a multiple of 256.
    const uint32_t qwen38_n_ctx_train    = 262144;
    const uint32_t gptoss_n_ctx_train    = 131072;
    const uint32_t unaligned_n_ctx_train = 40000;

    expect_resolution(0, qwen38_n_ctx_train, 0, false, "no request holds no room");
    expect_resolution(4096, qwen38_n_ctx_train, 4096, false, "an explicit -c 4096 is the room");
    expect_resolution(4000, qwen38_n_ctx_train, 4096, false, "an explicit hint is padded as llama_context pads n_ctx");
    expect_resolution(4096, 0, 4096, false, "an explicit hint does not depend on n_ctx_train");
    expect_resolution(262144, qwen38_n_ctx_train, 262144, false,
                      "an explicit -c equal to n_ctx_train is still the caller's, not the default");
    expect_resolution(524288, qwen38_n_ctx_train, 524288, false, "an explicit hint above n_ctx_train is honoured");

    expect_resolution(LLAMA_N_CTX_HINT_DEFAULT, qwen38_n_ctx_train, 262144, true,
                      "the default context of a Qwen3.8-like model is its training context");
    expect_resolution(LLAMA_N_CTX_HINT_DEFAULT, gptoss_n_ctx_train, 131072, true,
                      "the default context of a GPT-OSS-like model is its training context");
    expect_resolution(LLAMA_N_CTX_HINT_DEFAULT, unaligned_n_ctx_train, 40192, true,
                      "the default context is padded to 256 as llama_context pads n_ctx 0");
    expect_resolution(LLAMA_N_CTX_HINT_DEFAULT, 0, 0, false,
                      "a model without a training context holds no room for its default");

    if (g_failures == 0) {
        std::printf("test-n-ctx-hint-resolve: PASS\n");
    }
    return g_failures == 0 ? 0 : 1;
}
