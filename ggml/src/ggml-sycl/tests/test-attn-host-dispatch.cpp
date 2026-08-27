#include "../attn-host-dispatch.hpp"
#include "ggml.h"

#include <cstdio>

// The build is -DNDEBUG (Release), so assert() would compile away and the
// test would pass vacuously (llama.cpp-u2mz). Use an explicit check that
// always runs, per the test-kv-runtime-demotion.cpp / test-zone-sizing.cpp
// precedent in this directory.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::attn_op_consumes_tensor;
using ggml_sycl::attn_tensor_depends_on;

int main() {
    ggml_init_params params{ /*.mem_size   =*/16 * 1024 * 1024,
                             /*.mem_buffer =*/nullptr,
                             /*.no_alloc   =*/true };
    ggml_context *   ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init failed");

    // 1. null safety: neither function may dereference a null argument.
    {
        CHECK(!attn_tensor_depends_on(nullptr, nullptr), "case 1: null/null depends_on is false");
        CHECK(!attn_op_consumes_tensor(nullptr, nullptr), "case 1: null/null consumes is false");
        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        CHECK(!attn_tensor_depends_on(a, nullptr), "case 1: null target is false");
        CHECK(!attn_tensor_depends_on(nullptr, a), "case 1: null tensor is false");
    }

    // 2. direct producer/consumer: a MUL_MAT-shaped op with pending_dst as
    // src[0] must be recognized as a real consumer.
    {
        ggml_tensor * pending  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * consumer = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        consumer->src[0]       = pending;
        CHECK(attn_op_consumes_tensor(consumer, pending), "case 2: direct src[0] is a real consumer");
    }

    // 3. non-consumer: an unrelated op with no path to pending_dst must not
    // trigger a flush -- this is the case that must stay non-blocking for
    // genuine overlap.
    {
        ggml_tensor * pending   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * unrelated = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * other_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        unrelated->src[0]       = other_src;
        CHECK(!attn_op_consumes_tensor(unrelated, pending), "case 3: unrelated op is not a consumer");
    }

    // 4. transitive dependency via an intermediate op: consumer reads B,
    // which itself reads pending -- attn_tensor_depends_on must walk src[]
    // recursively, not just one hop.
    {
        ggml_tensor * pending      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * intermediate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        intermediate->src[0]       = pending;
        ggml_tensor * consumer     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        consumer->src[0]           = intermediate;
        CHECK(attn_op_consumes_tensor(consumer, pending), "case 4: transitive src[] dependency is a real consumer");
    }

    // 5. view relationship: a consumer reading a VIEW over pending_dst must
    // still be recognized -- KV consumers read a view over the cache tensor,
    // not the cache tensor itself (mirrors ggml_sycl_tensor_is_in_kv_host_buft's
    // view_src hop, ggml-sycl.cpp:99075-99084).
    {
        ggml_tensor * pending  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * view     = ggml_view_1d(ctx, pending, 4, 0);
        ggml_tensor * consumer = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        consumer->src[0]       = view;
        CHECK(attn_op_consumes_tensor(consumer, pending),
              "case 5: consumer of a view over pending_dst is a real consumer");
    }

    // 6. producer-writes-same-tensor is NOT a consumer: a fused op that
    // reuses pending_dst as its OWN dst (consuming_dst == pending_dst) must
    // not be treated as reading it -- mirrors the "fused MoE producer" skip
    // in ggml_sycl_op_consumes_tensor (ggml-sycl.cpp ~20182).
    // attn_op_consumes_tensor only scans consuming_dst->src[], so passing
    // pending_dst itself as consuming_dst, with no src[] pointing at it,
    // must return false.
    {
        ggml_tensor * pending = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        CHECK(!attn_op_consumes_tensor(pending, pending), "case 6: a node is not its own consumer");
    }

    // 7. depth bound: a src[] self-cycle must not recurse unboundedly (the
    // depth>32 guard). A well-formed ggml graph is acyclic, but the check
    // must not hang if one is ever malformed.
    {
        ggml_tensor * pending = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * cyclic  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        cyclic->src[0]        = cyclic;  // self-cycle, never reaches pending
        CHECK(!attn_tensor_depends_on(cyclic, pending), "case 7: bounded recursion terminates without finding target");
    }

    ggml_free(ctx);
    std::printf("test-attn-host-dispatch: all ok\n");
    return 0;
}
