#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdint>

// The context a load holds KV room for, resolved from llama_model_params::n_ctx_hint (llama.cpp-ak0p,
// llama.cpp-7fmt). This is the one place the hint is resolved: src/llama-model.cpp copies the result into the SYCL
// inventory, and the backend never sees the hint itself.
struct llama_n_ctx_hint_resolution {
    uint32_t n_ctx_context;  // cells the room is for; 0 = no room
    bool     is_default;     // n_ctx_context is the model's default context, not one the caller named
};

// LLAMA_N_CTX_HINT_DEFAULT is the model's default context: the one llama_context creates for n_ctx 0, which it
// resolves to n_ctx_train and pads to 256 (src/llama-context.cpp), for a single sequence. A non-unified multi-slot
// context pads each slot's share and can exceed it; the runtime transaction re-places that overflow. Read as a literal
// context the sentinel would pad, wrapping, to 0 and hold no room. 0 is no request: no room. Any other hint is the
// caller's own context, padded the same way; one above n_ctx_train (a rope-extended -c) is honoured.
static inline llama_n_ctx_hint_resolution llama_n_ctx_hint_resolve(uint32_t n_ctx_hint, uint32_t n_ctx_train) {
    if (n_ctx_hint == LLAMA_N_CTX_HINT_DEFAULT) {
        const uint32_t n_ctx_context = GGML_PAD(n_ctx_train, 256);
        return { n_ctx_context, n_ctx_context != 0 };
    }
    return { n_ctx_hint != 0 ? GGML_PAD(n_ctx_hint, 256) : 0, false };
}
