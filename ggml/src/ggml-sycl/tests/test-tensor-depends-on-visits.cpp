#include "../attn-host-dispatch.hpp"
#include "ggml.h"

#include <cstdio>
#include <set>
#include <vector>

// The build is -DNDEBUG, so assert() would compile away; use an explicit check.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

// Reference: the original path-by-path recursion, exhaustive, counting every
// call. Its count is the cost the production walk used to pay.
static bool reference_depends_on(const ggml_tensor * tensor, const ggml_tensor * target, int depth, size_t & calls) {
    if (!tensor || !target || depth > 32) {
        return false;
    }
    ++calls;
    for (const ggml_tensor * t = tensor; t; t = t->view_src) {
        if (t == target) {
            return true;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (reference_depends_on(tensor->src[i], target, depth + 1, calls)) {
            return true;
        }
    }
    return false;
}

static ggml_tensor * node(ggml_context * ctx) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
}

int main() {
    ggml_init_params params{ /*.mem_size =*/16 * 1024 * 1024, /*.mem_buffer =*/nullptr, /*.no_alloc =*/true };
    ggml_context *   ctx = ggml_init(params);
    CHECK(ctx != nullptr, "ggml_init failed");

    // Residual stream: x[i+1] = add(x[i], f[i]) with f[i] = g(x[i], x[i]).
    // x[i] is reachable from x[i+1] through three edges (add src0, and both
    // srcs of f), so the number of paths from x[n] back to x[0] is 3^n.
    auto stream = [&](int n_layers) {
        std::vector<ggml_tensor *> x;
        x.push_back(node(ctx));
        for (int i = 0; i < n_layers; ++i) {
            ggml_tensor * f = node(ctx);
            f->src[0]       = x.back();
            f->src[1]       = x.back();
            ggml_tensor * a = node(ctx);
            a->src[0]       = x.back();
            a->src[1]       = f;
            x.push_back(a);
        }
        return x;
    };
    const int                  n_layers = 64;
    std::vector<ggml_tensor *> x        = stream(n_layers);
    ggml_tensor *              root     = x.back();
    const size_t               n_nodes  = x.size() * 2 - 1;  // x[0] plus (f, add) per layer

    // 1. target reachable near the root: both agree.
    {
        const ggml_tensor * target    = x[n_layers - 3];
        size_t              ref_calls = 0, visits = 0;
        const bool          ref = reference_depends_on(root, target, 0, ref_calls);
        const bool          got = ggml_sycl::attn_tensor_depends_on_counted(root, target, 0, &visits);
        CHECK(ref && got, "case 1: near target reachable");
        std::printf("case 1 (reachable near root): reference calls=%zu, visits=%zu\n", ref_calls, visits);
        CHECK(visits <= n_nodes, "case 1: each node expanded at most once");
    }

    // 2. target unreachable: forces the full walk. The reference is only
    // runnable on a short stream (3^12 paths); the 64-layer stream checks the
    // production walk alone, where the depth cap bounds the visits.
    {
        std::vector<ggml_tensor *> xs        = stream(12);
        ggml_tensor *              other     = node(ctx);
        size_t                     ref_calls = 0, visits = 0;
        const bool                 ref = reference_depends_on(xs.back(), other, 0, ref_calls);
        const bool                 got = ggml_sycl::attn_tensor_depends_on_counted(xs.back(), other, 0, &visits);
        CHECK(!ref && !got, "case 2: unreachable target is not a dependency");
        std::printf("case 2 (unreachable, 12 layers): reference calls=%zu, visits=%zu\n", ref_calls, visits);
        CHECK(visits <= xs.size() * 2 - 1, "case 2: each node expanded at most once");

        size_t visits64 = 0;
        CHECK(!ggml_sycl::attn_tensor_depends_on_counted(root, other, 0, &visits64), "case 2: 64-layer unreachable");
        std::printf("case 2 (unreachable, 64 layers): visits=%zu\n", visits64);
        CHECK(visits64 <= n_nodes, "case 2: 64-layer visits at most once per node");
        CHECK(visits64 <= 70, "case 2: visits bounded by the depth cap, not by the path count");
    }

    // 3. target reachable only through a view_src chain.
    {
        ggml_tensor * base = x[n_layers - 5];
        ggml_tensor * v1   = ggml_view_1d(ctx, base, 4, 0);
        ggml_tensor * v2   = ggml_view_1d(ctx, v1, 2, 0);
        ggml_tensor * c    = node(ctx);
        c->src[0]          = v2;
        ggml_tensor * top  = node(ctx);
        top->src[0]        = x[n_layers - 4];
        top->src[1]        = c;
        size_t ref_calls = 0, visits = 0;
        CHECK(reference_depends_on(top, base, 0, ref_calls), "case 3: reference sanity");
        CHECK(ggml_sycl::attn_tensor_depends_on_counted(top, base, 0, &visits),
              "case 3: view_src chain reaches target");
        CHECK(ggml_sycl::attn_tensor_depends_on(top, base), "case 3: plain entry point agrees");
    }

    // 4. null arguments, and the depth cap itself: a target 40 hops down a
    // chain is beyond the cap for both walks.
    {
        size_t visits = 99;
        CHECK(!ggml_sycl::attn_tensor_depends_on_counted(nullptr, root, 0, &visits) && visits == 0,
              "case 4: null tensor");
        CHECK(!ggml_sycl::attn_tensor_depends_on_counted(root, nullptr, 0, &visits), "case 4: null target");
        CHECK(!ggml_sycl::attn_tensor_depends_on_counted(root, root->src[0], 33, &visits),
              "case 4: starting past the cap");

        std::vector<ggml_tensor *> chain;
        chain.push_back(node(ctx));
        for (int i = 0; i < 40; ++i) {
            ggml_tensor * n = node(ctx);
            n->src[0]       = chain.back();
            chain.push_back(n);
        }
        size_t     ref_calls = 0;
        const bool ref       = reference_depends_on(chain.back(), chain[0], 0, ref_calls);
        const bool got       = ggml_sycl::attn_tensor_depends_on(chain.back(), chain[0]);
        CHECK(ref == got, "case 4: cap behaviour matches the reference");
        CHECK(ggml_sycl::attn_tensor_depends_on(chain.back(), chain[40 - 32]), "case 4: exactly at the cap is found");
        CHECK(!ggml_sycl::attn_tensor_depends_on(chain.back(), chain[40 - 33]), "case 4: one past the cap is not");
    }

    // 5. a src[] self-cycle terminates.
    {
        ggml_tensor * cyc = node(ctx);
        cyc->src[0]       = cyc;
        CHECK(!ggml_sycl::attn_tensor_depends_on(cyc, node(ctx)), "case 5: cycle terminates");
    }

    // 6. consumer entry point: srcs only, never the node itself.
    {
        CHECK(ggml_sycl::attn_op_consumes_tensor(root, x[n_layers - 10]), "case 6: root consumes an early layer");
        CHECK(!ggml_sycl::attn_op_consumes_tensor(x[n_layers - 10], x[n_layers - 10]),
              "case 6: a node is not its own consumer");
    }

    // 7. consumer entry point: a residual block's two srcs (x[n-1] and f)
    // share their whole upstream subgraph, so one walk must expand each
    // distinct node once, not once per src.
    {
        ggml_tensor * block  = node(ctx);
        block->src[0]        = x[n_layers];
        block->src[1]        = x[n_layers]->src[1];  // f of the last layer, itself fed by x[n_layers - 1]
        ggml_tensor * other  = node(ctx);
        size_t        visits = 0, per_src = 0, v = 0;
        CHECK(!ggml_sycl::attn_op_consumes_tensor_counted(block, other, &visits), "case 7: unreachable target");
        for (int i = 0; i < 2; ++i) {
            ggml_sycl::attn_tensor_depends_on_counted(block->src[i], other, 0, &v);
            per_src += v;
        }
        std::printf("case 7 (consumer, two shared srcs): one walk visits=%zu, per-src walks=%zu\n", visits, per_src);
        // Bound by the distinct nodes within 32 levels of the srcs, not by the
        // whole graph: the cap hides the deepest layers, so n_nodes would let a
        // per-src walk (which re-expands the shared part) pass.
        std::set<const ggml_tensor *>    seen;
        std::vector<const ggml_tensor *> level;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (block->src[i] && seen.insert(block->src[i]).second) {
                level.push_back(block->src[i]);
            }
        }
        for (int d = 0; d <= 32 && !level.empty(); ++d) {
            std::vector<const ggml_tensor *> next;
            for (const ggml_tensor * t : level) {
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (t->src[i] && seen.insert(t->src[i]).second) {
                        next.push_back(t->src[i]);
                    }
                }
            }
            level.swap(next);
        }
        // Nodes first reached at level 33 are never expanded, so drop them.
        const size_t reachable = seen.size() - level.size();
        std::printf("case 7: distinct reachable within the cap=%zu\n", reachable);
        CHECK(visits <= reachable, "case 7: consumer query expands each reachable node at most once");
        CHECK(visits < per_src, "case 7: shared subgraph is not re-expanded per src");
        CHECK(ggml_sycl::attn_op_consumes_tensor_counted(block, x[n_layers - 4], &visits), "case 7: reachable target");
        CHECK(!ggml_sycl::attn_op_consumes_tensor_counted(nullptr, other, &visits) && visits == 0,
              "case 7: null consumer");
    }

    ggml_free(ctx);
    std::printf("test-tensor-depends-on-visits: all ok\n");
    return 0;
}
