// Decode/prompt classification of a backend-scheduler split. A split is
// classified from its batch evidence: a dense MUL_MAT's src1 rows or a
// MUL_MAT_ID's routed-token count. ggml-sycl/graph-phase.hpp is SYCL-free, so
// this builds synthetic graphs on a no-alloc ggml context and needs no device.
#include "ggml-sycl/graph-phase.hpp"

#include <cstdio>

#define CHECK(cond, msg)                      \
    do {                                      \
        if (!(cond)) {                        \
            std::printf("FAILED: %s\n", msg); \
            return 1;                         \
        }                                     \
    } while (0)

namespace {

struct graph_fixture {
    ggml_context * ctx   = nullptr;
    ggml_cgraph *  graph = nullptr;

    graph_fixture() {
        ggml_init_params params = {};
        params.mem_size         = 16 * 1024 * 1024;
        params.no_alloc         = true;
        ctx                     = ggml_init(params);
        graph                   = ggml_new_graph(ctx);
    }

    ~graph_fixture() { ggml_free(ctx); }

    // A dense projection of n_tokens rows.
    void add_dense(int64_t n_tokens) {
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, n_tokens);
        ggml_build_forward_expand(graph, ggml_mul_mat(ctx, w, x));
    }

    // A routed expert projection: 8 experts, 2 used per token, n_tokens tokens.
    void add_moe(int64_t n_tokens) {
        ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 32, 8);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, n_tokens);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_tokens);
        ggml_build_forward_expand(graph, ggml_mul_mat_id(ctx, w, x, ids));
    }

    // Neither kind of matmul: a norm and an add over n_tokens rows.
    void add_elementwise(int64_t n_tokens) {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, n_tokens);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, n_tokens);
        ggml_build_forward_expand(graph, ggml_add(ctx, ggml_rms_norm(ctx, a, 1e-6f), b));
    }

    bool is_decode(bool previous) const {
        ggml_tensor * nodes[64];
        const int     n = ggml_graph_n_nodes(graph);
        if (n > 64) {
            return !previous;  // the fixture never builds this many nodes; make it visible
        }
        for (int i = 0; i < n; ++i) {
            nodes[i] = ggml_graph_node(graph, i);
        }
        return ggml_sycl::graph_phase_is_decode(nodes, n, previous);
    }
};

int test_dense_evidence() {
    {
        graph_fixture g;
        g.add_elementwise(1);
        g.add_dense(1);
        CHECK(g.is_decode(false), "a split whose dense MUL_MAT has one row is decode, whatever came before");
    }
    {
        graph_fixture g;
        g.add_dense(7);
        CHECK(!g.is_decode(true), "a split whose dense MUL_MAT has several rows is a prompt, whatever came before");
    }
    return 0;
}

int test_moe_evidence() {
    {
        graph_fixture g;
        g.add_elementwise(1);
        g.add_moe(1);
        CHECK(g.is_decode(false), "a split with only a batch-1 MUL_MAT_ID is decode");
    }
    {
        graph_fixture g;
        g.add_moe(5);
        CHECK(!g.is_decode(true), "a split with only a MUL_MAT_ID routing several tokens is a prompt");
    }
    return 0;
}

int test_first_evidence_decides() {
    graph_fixture g;
    g.add_moe(1);
    g.add_dense(1);
    CHECK(g.is_decode(false), "the first matmul in node order decides (MUL_MAT_ID first, decode)");
    return 0;
}

int test_no_evidence_keeps_phase() {
    {
        graph_fixture g;
        g.add_elementwise(1);
        CHECK(g.is_decode(true), "a split with no matmul keeps the decode phase it was in");
    }
    {
        graph_fixture g;
        g.add_elementwise(9);
        CHECK(!g.is_decode(false), "a split with no matmul keeps the prompt phase it was in");
    }
    {
        graph_fixture g;
        CHECK(g.is_decode(true), "an empty split keeps the decode phase it was in");
    }
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= test_dense_evidence();
    rc |= test_moe_evidence();
    rc |= test_first_evidence_decides();
    rc |= test_no_evidence_keeps_phase();
    if (rc == 0) {
        std::printf("test-sycl-graph-phase: all checks passed\n");
    }
    return rc;
}
