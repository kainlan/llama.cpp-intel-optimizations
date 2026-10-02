#include "../context-tenant-measure.hpp"
#include "../nonfa-stage.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

// Release builds define NDEBUG, which would compile assert() away and let this
// test pass vacuously, so every check is an explicit one that always runs.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

using ggml_sycl::context_demand_accum;
using ggml_sycl::context_measure_view;

namespace {

// The environment the visitor's view hands back, set per case.
ggml_sycl_mul_mat_route_env g_env       = {};
int                         g_env_calls = 0;

bool env_of(void *, const ggml_tensor *, ggml_sycl_mul_mat_route_env * out) {
    g_env_calls++;
    *out = g_env;
    return true;
}

// The shapes below are the ones llama.cpp's non-FA attention builds: Q and K are
// permuted views, V is the transposed cache, and the softmax is a contiguous f32.
struct attn {
    ggml_context * ctx;
    int64_t        d, n_head, n_head_kv, n_kv, n_ub, n_seq;
};

// KQ = mul_mat(K, Q): K the permuted cache view, Q the permuted query.
ggml_tensor * make_kq(const attn & a) {
    ggml_tensor * k_cache = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F16, a.d, a.n_head_kv, a.n_kv, a.n_seq);
    ggml_tensor * k       = ggml_permute(a.ctx, k_cache, 0, 2, 1, 3);
    ggml_tensor * q_in    = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, a.d, a.n_head, a.n_ub, a.n_seq);
    ggml_tensor * q       = ggml_permute(a.ctx, q_in, 0, 2, 1, 3);
    ggml_tensor * kq      = ggml_mul_mat(a.ctx, k, q);
    ggml_set_name(kq, "kq-0");
    return kq;
}

// KQV = mul_mat(V, softmax): V the transposed cache. `kv_size` > n_kv makes the
// view non-contiguous, as a cache that is not full is.
ggml_tensor * make_kqv(const attn & a, int64_t kv_size) {
    ggml_tensor * v_cache = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F16, kv_size, a.d, a.n_head_kv, a.n_seq);
    ggml_tensor * v    = ggml_view_4d(a.ctx, v_cache, a.n_kv, a.d, a.n_head_kv, a.n_seq, v_cache->nb[1], v_cache->nb[2],
                                      v_cache->nb[3], 0);
    ggml_tensor * soft = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, a.n_kv, a.n_ub, a.n_head, a.n_seq);
    ggml_tensor * kqv  = ggml_mul_mat(a.ctx, v, soft);
    ggml_set_name(kqv, "kqv-0");
    return kqv;
}

ggml_sycl_mul_mat_route_env env(bool split, bool has_weight, bool force_simple, bool strided) {
    ggml_sycl_mul_mat_route_env e = {};
    e.split                       = split;
    e.has_weight                  = has_weight;
    e.kqv_force_simple            = force_simple;
    e.stage_strided               = strided;
    return e;
}

uint64_t visit_bytes(const ggml_tensor * node, const ggml_sycl_mul_mat_route_env & e, bool * ok) {
    g_env = e;
    context_demand_accum acc;
    context_measure_view view;
    view.device            = 0;
    view.mul_mat_route_env = env_of;
    ggml_sycl::context_nonfa_stage_visit(node, view, acc);
    *ok          = acc.ok();
    const auto t = acc.tenants();
    if (t.empty()) {
        return 0;
    }
    if (t.size() != 1 || t[0].cohort != GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE || t[0].slot_index != 0 ||
        t[0].device != 0) {
        *ok = false;
    }
    return t[0].slot_bytes;
}

}  // namespace

int main() {
    ggml_init_params ip  = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
    ggml_context *   ctx = ggml_init(ip);
    CHECK(ctx != nullptr, "ggml_init");

    attn                              a     = { ctx, 128, 8, 2, 256, 16, 1 };
    const ggml_sycl_mul_mat_route_env plain = env(false, false, false, true);

    // ---- the route classifier: the same chain ggml_sycl_mul_mat walks ----
    ggml_tensor * kq_pp = make_kq(a);  // n_ub 16: not the single-token branch
    CHECK(ggml_sycl_mul_mat_f16_route_of(kq_pp->src[0], kq_pp->src[1], kq_pp, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED,
          "PP KQ is the multi-batch batched route");
    CHECK(ggml_sycl_mul_mat_routes_batched_f16(kq_pp->src[0], kq_pp->src[1], kq_pp, plain), "PP KQ routes batched");

    attn a1             = a;
    a1.n_ub             = 1;
    ggml_tensor * kq_tg = make_kq(a1);  // one token, one sequence: the p021 kernel stages nothing
    CHECK(ggml_sycl_mul_mat_f16_route_of(kq_tg->src[0], kq_tg->src[1], kq_tg, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_P021,
          "single-token single-sequence KQ is the p021 kernel");
    CHECK(!ggml_sycl_mul_mat_routes_batched_f16(kq_tg->src[0], kq_tg->src[1], kq_tg, plain),
          "the p021 kernel is not the batched route");

    attn a2              = a1;
    a2.n_seq             = 2;
    ggml_tensor * kq_tg2 = make_kq(a2);  // two sequences: the p021 kernel does not support it, batched does
    CHECK(ggml_sycl_mul_mat_f16_route_of(kq_tg2->src[0], kq_tg2->src[1], kq_tg2, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQ_BATCHED,
          "single-token two-sequence KQ is the batched route of the p021 branch");

    ggml_tensor * kqv_pp = make_kqv(a, 256);
    CHECK(ggml_sycl_mul_mat_f16_route_of(kqv_pp->src[0], kqv_pp->src[1], kqv_pp, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED,
          "PP KQV is the multi-batch batched route");

    ggml_tensor * kqv_tg_full = make_kqv(a1, 256);  // a full cache is contiguous: not the nc kernel
    CHECK(ggml_sycl_mul_mat_routes_batched_f16(kqv_tg_full->src[0], kqv_tg_full->src[1], kqv_tg_full, plain),
          "single-token KQV over a contiguous cache is batched");

    ggml_tensor * kqv_tg_part = make_kqv(a1, 512);  // a partly full cache is a strided view: the nc kernel
    CHECK(ggml_sycl_mul_mat_f16_route_of(kqv_tg_part->src[0], kqv_tg_part->src[1], kqv_tg_part, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_VEC_NC,
          "single-token KQV over a strided view is the nc kernel");
    CHECK(!ggml_sycl_mul_mat_routes_batched_f16(kqv_tg_part->src[0], kqv_tg_part->src[1], kqv_tg_part, plain),
          "the nc kernel is not the batched route");

    // The nc kernel is the single-sequence case; two sequences fall through to batched.
    ggml_tensor * kqv_tg_part2 = make_kqv(a2, 512);
    CHECK(ggml_sycl_mul_mat_f16_route_of(kqv_tg_part2->src[0], kqv_tg_part2->src[1], kqv_tg_part2, plain) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_BATCHED,
          "single-token KQV over a strided view of two sequences is batched, not the nc kernel");

    // Each term of the predicate takes the node off the route.
    CHECK(!ggml_sycl_mul_mat_routes_batched_f16(kq_pp->src[0], kq_pp->src[1], kq_pp, env(true, false, false, true)),
          "a row-split src0 never routes batched");
    CHECK(!ggml_sycl_mul_mat_routes_batched_f16(kq_pp->src[0], kq_pp->src[1], kq_pp, env(false, true, false, true)),
          "a weight operand never routes batched");
    CHECK(!ggml_sycl_mul_mat_routes_batched_f16(kqv_pp->src[0], kqv_pp->src[1], kqv_pp, env(false, true, false, true)),
          "a weight operand never routes batched (KQV)");
    CHECK(ggml_sycl_mul_mat_f16_route_of(kqv_pp->src[0], kqv_pp->src[1], kqv_pp, env(false, false, true, true)) ==
              GGML_SYCL_MUL_MAT_F16_ROUTE_KQKV_SCALAR,
          "the KQV debug override takes a kqv node to the scalar fallback");
    CHECK(ggml_sycl_mul_mat_routes_batched_f16(kq_pp->src[0], kq_pp->src[1], kq_pp, env(false, false, true, true)),
          "the KQV debug override does not touch a node that is not kqv");

    // A mul_mat whose src0 is not f16 is never on this route, whatever the env.
    {
        ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 4, 8);
        ggml_tensor * out = ggml_mul_mat(ctx, w, x);
        CHECK(ggml_sycl_mul_mat_f16_route_of(w, x, out, plain) == GGML_SYCL_MUL_MAT_F16_ROUTE_NONE,
              "a quantized src0 is not an f16 route");
    }

    // ---- the staging size: what alloc() receives at the site ----
    // The contiguous Q of KQ: strided and element counts agree.
    CHECK(ggml_sycl_batched_f16_src1_stage_elems(kq_pp->src[1], true) == 128u * 8u * 16u, "Q strided elems");
    CHECK(ggml_sycl_batched_f16_src1_stage_elems(kq_pp->src[1], false) == 128u * 8u * 16u, "Q element count");
    // The GI4 softmax: 34816 x 512 x 32 f32, staged as f16.
    {
        attn          gi4 = { ctx, 128, 32, 8, 34816, 512, 1 };
        ggml_tensor * kqv = make_kqv(gi4, 34816);
        CHECK(ggml_sycl_batched_f16_src1_stage_elems(kqv->src[1], true) == 34816ull * 512ull * 32ull,
              "GI4 softmax elems");
        bool           ok    = false;
        const uint64_t bytes = visit_bytes(kqv, plain, &ok);
        CHECK(ok && bytes == 1140850688ull, "GI4 stages 1088 MiB, the design's number");
    }
    // An f16 src1 is not staged.
    {
        ggml_tensor * v_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 128, 2, 1);
        ggml_tensor * p       = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 16, 8, 1);
        ggml_tensor * out     = ggml_mul_mat(ctx, v_cache, p);
        ggml_set_name(out, "kqv-f16");
        CHECK(ggml_sycl_batched_f16_src1_stage_elems(p, true) == 0 &&
                  ggml_sycl_batched_f16_src1_stage_elems(p, false) == 0,
              "an f16 src1 stages nothing");
        bool ok = false;
        CHECK(visit_bytes(out, plain, &ok) == 0 && ok, "the visitor demands nothing for an f16 src1");
    }
    // A strided src1 view: the oneDNN-strided path stages nb[last]*ne[last], the
    // oneMath path stages the element count, and they differ. This is the case
    // where a visitor that counted one and a site that staged the other would
    // disagree.
    ggml_tensor * strided_out = nullptr;
    {
        ggml_tensor * parent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 128);  // rows of 256, view takes 128 of each
        ggml_tensor * src1   = ggml_view_3d(ctx, parent, 128, 16, 8, parent->nb[1], parent->nb[1] * 16, 0);
        ggml_tensor * k_c    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 256, 8, 1);
        strided_out          = ggml_mul_mat(ctx, k_c, src1);
        ggml_set_name(strided_out, "kq-strided");
        CHECK(src1->nb[1] == 1024 && src1->nb[3] == 131072, "the strided view is shaped as the case assumes");
        CHECK(ggml_sycl_batched_f16_src1_stage_elems(src1, true) == 32768, "strided path: nb[last]*ne[last]/4");
        CHECK(ggml_sycl_batched_f16_src1_stage_elems(src1, false) == 16384, "oneMath path: element count");
        bool ok = false;
        CHECK(visit_bytes(strided_out, env(false, false, false, true), &ok) == 65536 && ok,
              "the visitor counts the strided staging when the site will stage it");
        CHECK(visit_bytes(strided_out, env(false, false, false, false), &ok) == 32768 && ok,
              "and the element count when the site will stage that");
    }

    // ---- the visitor counts exactly what the routing selects ----
    {
        std::vector<ggml_tensor *> nodes = { kq_pp, kq_tg, kq_tg2, kqv_pp, kqv_tg_full, kqv_tg_part, strided_out };
        const ggml_sycl_mul_mat_route_env envs[] = {
            env(false, false, false, true), env(false, false, false, false), env(true, false, false, true),
            env(false, true, false, true),  env(false, false, true, true),   env(false, false, true, false),
        };
        for (const auto & e : envs) {
            for (ggml_tensor * n : nodes) {
                const bool     routed = ggml_sycl_mul_mat_routes_batched_f16(n->src[0], n->src[1], n, e);
                const uint64_t want =
                    routed ? 2ull * ggml_sycl_batched_f16_src1_stage_elems(n->src[1], e.stage_strided) : 0ull;
                bool           ok  = false;
                const uint64_t got = visit_bytes(n, e, &ok);
                CHECK(ok, "the visitor records no error for a sized node");
                CHECK(got == want, "the visitor's demand is the routing's elems at 2 B, or nothing");
            }
        }
    }

    // ---- the accumulator keeps the maximum over the visited nodes ----
    {
        g_env = plain;
        context_demand_accum acc;
        context_measure_view view;
        view.device            = 1;
        view.mul_mat_route_env = env_of;
        ggml_sycl::context_nonfa_stage_visit(kq_pp, view, acc);   // 128*8*16*2 = 32768 B
        ggml_sycl::context_nonfa_stage_visit(kqv_pp, view, acc);  // 256*16*8*2 = 65536 B
        ggml_sycl::context_nonfa_stage_visit(kq_pp, view, acc);   // a smaller one after: no change
        const auto t = acc.tenants();
        CHECK(acc.ok() && t.size() == 1, "one slot over the visited nodes");
        CHECK(t[0].cohort == GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE && t[0].slot_index == 0 && t[0].device == 1 &&
                  t[0].slot_bytes == 65536,
              "the slot is the maximum of the nodes' staging, on the view's device");
    }

    // ---- a node that is not a mul_mat, or not f16 src0, needs no environment ----
    {
        context_demand_accum acc;
        context_measure_view view;  // no mul_mat_route_env
        ggml_tensor *        add =
            ggml_add(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8));
        ggml_sycl::context_nonfa_stage_visit(add, view, acc);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 128, 64);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 4, 8);
        ggml_sycl::context_nonfa_stage_visit(ggml_mul_mat(ctx, w, x), view, acc);
        CHECK(acc.ok() && acc.tenants().empty(), "an op with no f16 route is not an error without an environment");

        // An f16 mul_mat cannot be classified without the environment, and an
        // unsized demand is not a zero.
        ggml_sycl::context_nonfa_stage_visit(kq_pp, view, acc);
        CHECK(!acc.ok() && acc.error().find("context-nonfa-stage") != std::string::npos,
              "an f16 mul_mat with no environment is a named error");
    }

    // ---- the registered table carries the visitor ----
    {
        const ggml_sycl::context_measure_visitor * t     = ggml_sycl::context_measure_visitors();
        bool                                       found = false;
        for (size_t i = 0; t[i].fn != nullptr && i < 64; i++) {
            if (t[i].fn == ggml_sycl::context_nonfa_stage_visit) {
                found = true;
            }
        }
        CHECK(found, "the nonfa-stage visitor is in the table");
    }

    ggml_free(ctx);
    std::printf("PASS\n");
    return 0;
}
