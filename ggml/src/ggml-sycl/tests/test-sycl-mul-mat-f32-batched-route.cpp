// Host test of the batched F32 MUL_MAT route's predicates (nonfa-stage.hpp): which F32 x F32 MUL_MAT nodes the backend
// runs as one oneDNN matmul, and which it leaves to the per-plane loop of ggml_sycl_op_mul_mat.
//
// The route is two pure functions: ggml_sycl_mul_mat_f32_batched_route_of reads the three tensors only, and
// ggml_sycl_mul_mat_f32_batched_env_admits reads the environment only. The dispatch asks the tensors first and builds
// the environment (a queue-state query among its facts) only for a node they admit. Every accept and decline
// condition is one case here, with the tensors built by ggml itself in a no_alloc context (no device, no data).
// The three Inkling shapes are built at their model dimensions (n_embd 2304, n_head 24, n_relative 16, the sliding
// window's bias table 1024 + 1 rows, at most 4 earlier blocks and 3 queries in graph::mix) at T = 512.

#include "../nonfa-stage.hpp"

#include <cstdint>
#include <cstdio>

// Release builds define NDEBUG, which would compile assert() away and let this
// test pass vacuously, so every check is an explicit one that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

namespace {

constexpr ggml_sycl_mul_mat_f32_batched_route NONE    = GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_NONE;
constexpr ggml_sycl_mul_mat_f32_batched_route COLUMNS = GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_COLUMNS;
constexpr ggml_sycl_mul_mat_f32_batched_route BATCHED = GGML_SYCL_MUL_MAT_F32_BATCHED_ROUTE_BATCHED;

// The environment the route admits: activations in this device's memory, no split, no tensor parallelism, no
// recording. Each environment case below changes one field.
ggml_sycl_mul_mat_f32_batched_env env_taken() {
    ggml_sycl_mul_mat_f32_batched_env env;
    env.on_device = true;
    return env;
}

ggml_sycl_mul_mat_f32_batched_route route_of(const ggml_tensor * dst) {
    return ggml_sycl_mul_mat_f32_batched_route_of(dst->src[0], dst->src[1], dst);
}

ggml_tensor * t4(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2 = 1, int64_t ne3 = 1) {
    return ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
}

int run(ggml_context * ctx) {
    // ---- taken ----

    // Equal batch in dim 2: one GEMM with batch dims.
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), t4(ctx, GGML_TYPE_F32, 64, 10, 6))) == BATCHED,
          "an equal batch in dim 2 is one batched GEMM");

    // Equal batch in dims 2 and 3: the planes of both are dense in the same order, so the batch is ne2 * ne3.
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 3, 2), t4(ctx, GGML_TYPE_F32, 64, 10, 3, 2))) ==
              BATCHED,
          "an equal batch in dims 2 and 3 is one batched GEMM");

    // A 2-D src0 against a src1 batched in dim 2: the batch folds into the columns of one plain GEMM.
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24), t4(ctx, GGML_TYPE_F32, 64, 10, 6))) == COLUMNS,
          "a 2-D src0 broadcast over dim 2 is one GEMM over every column");

    // Inkling graph::mix, scores = mul_mat(states [n_embd, nb+1, T], queries [n_embd, nq, T]).
    const int64_t T = 512;
    CHECK(
        route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 2304, 5, T), t4(ctx, GGML_TYPE_F32, 2304, 3, T))) == BATCHED,
        "Inkling mix scores: one batched GEMM");
    // Inkling graph::mix, mul_mat(cont(transpose(states)) [nb+1, n_embd, T], scores [nb+1, nq, T]).
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 5, 2304, T), t4(ctx, GGML_TYPE_F32, 5, 3, T))) == BATCHED,
          "Inkling mix output: one batched GEMM");
    // Inkling bias table, mul_mat(pad(attn_rel_bias) [n_rel, n_swa + 1], Rcur [n_rel, n_head, T]).
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 16, 1025), t4(ctx, GGML_TYPE_F32, 16, 24, T))) == COLUMNS,
          "Inkling bias table: one GEMM over every column");

    // A contiguous tensor ggml_is_permuted reports as permuted: [K, B, 1] permuted to [K, 1, B]. Its size-1 dim
    // carries a stride larger than the next dim's, which ggml_is_permuted compares, but no element sits anywhere
    // else than in the dense layout, which is all the route reads. Contiguity is the test, so it is taken.
    {
        ggml_tensor * base = t4(ctx, GGML_TYPE_F32, 64, 6, 1);
        ggml_tensor * src1 = ggml_permute(ctx, base, 0, 2, 1, 3);
        CHECK(ggml_is_permuted(src1) && ggml_is_contiguous(src1),
              "the case is a contiguous tensor reported as permuted");
        CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), src1)) == BATCHED,
              "a contiguous src1 that ggml_is_permuted reports as permuted is taken");
    }

    // ---- declined: the batch ----

    // Decode's T = 1 node: the tensor predicate alone, which takes no environment, declines it, so the dispatch never
    // builds an environment (nor queries the queue's state) for it.
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24), t4(ctx, GGML_TYPE_F32, 64, 10))) == NONE,
          "a batch of 1 is not this route's, decided from the tensors alone");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 2), t4(ctx, GGML_TYPE_F32, 64, 10, 6))) == NONE,
          "a partial broadcast in dim 2 (src0 batch 2 of 6) declines");
    CHECK(
        route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 1, 2), t4(ctx, GGML_TYPE_F32, 64, 10, 3, 2))) == NONE,
        "a broadcast in dim 2 under an equal dim 3 declines");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24), t4(ctx, GGML_TYPE_F32, 64, 10, 1, 4))) == NONE,
          "a 2-D src0 against a batch in dim 3 declines");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24), t4(ctx, GGML_TYPE_F32, 64, 10, 3, 2))) == NONE,
          "a 2-D src0 against a batch in dims 2 and 3 declines");

    // ---- declined: the types ----

    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F16, 64, 24, 6), t4(ctx, GGML_TYPE_F32, 64, 10, 6))) == NONE,
          "an F16 src0 declines (it is the f16 chain's)");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), t4(ctx, GGML_TYPE_F16, 64, 10, 6))) == NONE,
          "an F16 src1 declines");
    {
        ggml_tensor * dst = ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), t4(ctx, GGML_TYPE_F32, 64, 10, 6));
        dst->type         = GGML_TYPE_F16;  // ggml_mul_mat always makes an F32 dst; the predicate still asks
        CHECK(route_of(dst) == NONE, "a dst that is not F32 declines");
    }

    // ---- declined: contiguity ----

    {
        // src0 a view of every other row: rows are dense, planes are not.
        ggml_tensor * whole = t4(ctx, GGML_TYPE_F32, 64, 48, 6);
        ggml_tensor * src0  = ggml_view_3d(ctx, whole, 64, 24, 6, whole->nb[1] * 2, whole->nb[2], 0);
        CHECK(!ggml_is_contiguous(src0), "the src0 view is not contiguous");
        CHECK(route_of(ggml_mul_mat(ctx, src0, t4(ctx, GGML_TYPE_F32, 64, 10, 6))) == NONE,
              "a non-contiguous src0 declines");
    }
    {
        // src1 permuted with every dim above 1: [K, B, N] viewed as [K, N, B].
        ggml_tensor * src1 = ggml_permute(ctx, t4(ctx, GGML_TYPE_F32, 64, 6, 10), 0, 2, 1, 3);
        CHECK(!ggml_is_contiguous(src1), "the permuted src1 is not contiguous");
        CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), src1)) == NONE,
              "a non-contiguous src1 declines");
    }
    {
        ggml_tensor * dst = ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 64, 24, 6), t4(ctx, GGML_TYPE_F32, 64, 10, 6));
        dst->nb[2]        = dst->nb[2] * 2;  // planes apart: gemm writes C with a fixed plane stride
        dst->nb[3]        = dst->nb[2] * dst->ne[2];
        CHECK(!ggml_is_contiguous(dst), "the dst with spaced planes is not contiguous");
        CHECK(route_of(dst) == NONE, "a non-contiguous dst declines");
    }

    // ---- declined: the GEMM dimensions gemm() takes as int ----

    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 16, 1025),
                                t4(ctx, GGML_TYPE_F32, 16, 24, INT64_C(1) << 27))) == NONE,
          "a column count ne11 * ne12 above INT32_MAX declines");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, 4, INT64_C(1) << 31, 2),
                                t4(ctx, GGML_TYPE_F32, 4, 3, 2))) == NONE,
          "a row count ne01 above INT32_MAX declines");
    CHECK(route_of(ggml_mul_mat(ctx, t4(ctx, GGML_TYPE_F32, INT64_C(1) << 31, 2, 2),
                                t4(ctx, GGML_TYPE_F32, INT64_C(1) << 31, 3, 2))) == NONE,
          "a depth ne00 above INT32_MAX declines");

    // ---- the environment, one field each ----

    {
        CHECK(ggml_sycl_mul_mat_f32_batched_env_admits(env_taken()),
              "the environment cases start from an admitted one");

        ggml_sycl_mul_mat_f32_batched_env env = env_taken();
        env.split                             = true;
        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(env), "a src0 in a row-split buffer declines");

        env                 = env_taken();
        env.tensor_parallel = true;
        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(env), "tensor parallelism declines");

        env            = env_taken();
        env.has_weight = true;
        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(env), "an operand that is a weight declines");

        env           = env_taken();
        env.recording = true;
        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(env),
              "a command graph recording declines (a new key would JIT into it)");

        env           = env_taken();
        env.on_device = false;
        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(env), "an operand outside this device's memory declines");

        CHECK(!ggml_sycl_mul_mat_f32_batched_env_admits(ggml_sycl_mul_mat_f32_batched_env{}),
              "an environment nobody filled declines");
    }

    std::printf("test-sycl-mul-mat-f32-batched-route: PASS\n");
    return 0;
}

}  // namespace

int main() {
    ggml_init_params params = { /*.mem_size   =*/16 * 1024 * 1024, /*.mem_buffer =*/nullptr, /*.no_alloc   =*/true };
    ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }
    const int rc = run(ctx);
    ggml_free(ctx);
    return rc;
}
