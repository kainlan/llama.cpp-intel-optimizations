// llama_moe_profiler::flush() must stride each captured expert-id tensor by
// that tensor's own row width (ne[0]), not by the profile-wide n_expert_used,
// which since upstream made n_expert_used per layer is the MAXIMUM over all
// layers (llama.cpp-n77l). A layer that selects fewer experts than the maximum
// was read at the wide stride: past its own ids, and every token after the
// first at the wrong offset.
//
// Deterministic without reading out of bounds: layer 0 is captured at the
// maximum width and fills the profiler's read buffer with expert 7, then
// layer 1 is captured at half that width. The buffer is not shrunk, so a wide
// read of layer 1 picks up layer 0's stale 7s, and expert 7 shows up in a
// layer that never selected it.
//
// Host-only: the tensors live in a CPU buffer wrapped around a static array,
// so no backend registry, device or model is involved.

#include "../src/llama-moe-profile.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static int n_failures = 0;

static void check_eq(const char * what, uint64_t got, uint64_t want) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s = %llu, want %llu\n", what, (unsigned long long) got, (unsigned long long) want);
        n_failures++;
    }
}

int main() {
    const int n_expert     = 8;
    const int n_used_max   = 4;
    const int n_used_small = 2;
    const int n_tokens     = 2;

    ggml_init_params params = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc =*/true };
    ggml_context *   ctx    = ggml_init(params);

    ggml_tensor * wide   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used_max, n_tokens);
    ggml_tensor * narrow = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used_small, n_tokens);

    alignas(64) static uint8_t storage[256];
    ggml_backend_buffer_t      buf = ggml_backend_cpu_buffer_from_ptr(storage, sizeof(storage));
    if (ggml_backend_tensor_alloc(buf, wide, storage) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(buf, narrow, storage + 128) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: could not place the test tensors\n");
        return 1;
    }

    const int32_t wide_ids[n_used_max * n_tokens]     = { 7, 7, 7, 7, 7, 7, 7, 7 };
    const int32_t narrow_ids[n_used_small * n_tokens] = { 0, 1, 2, 3 };
    ggml_backend_tensor_set(wide, wide_ids, 0, sizeof(wide_ids));
    ggml_backend_tensor_set(narrow, narrow_ids, 0, sizeof(narrow_ids));

    llama_moe_profiler profiler;
    profiler.enabled = true;
    profiler.profile.init(/*n_layer =*/2, n_expert, n_used_max);

    profiler.schedule_capture(0, wide, n_tokens, n_used_max);
    profiler.schedule_capture(1, narrow, n_tokens, n_used_small);
    profiler.flush(nullptr);

    const llama_moe_layer_stats & l0 = profiler.profile.layer_stats[0];
    const llama_moe_layer_stats & l1 = profiler.profile.layer_stats[1];

    check_eq("layer 0 selections", l0.total_selections, n_used_max * n_tokens);
    check_eq("layer 0 expert 7", l0.expert_counts[7], n_used_max * n_tokens);

    check_eq("layer 1 selections", l1.total_selections, n_used_small * n_tokens);
    for (int e = 0; e < 4; e++) {
        char what[32];
        snprintf(what, sizeof(what), "layer 1 expert %d", e);
        check_eq(what, l1.expert_counts[e], 1);
    }
    check_eq("layer 1 expert 7 (layer 0's stale ids)", l1.expert_counts[7], 0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (n_failures) {
        fprintf(stderr, "%d check(s) failed\n", n_failures);
        return 1;
    }
    printf("OK: each captured layer is strided by its own row width\n");
    return 0;
}
