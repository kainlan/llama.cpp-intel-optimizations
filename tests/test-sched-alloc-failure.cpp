// A buffer type that refuses an allocation must reach the caller of
// ggml_backend_sched_alloc_graph as `false`, every time: the allocator's
// layout (node_allocs) must not outlive the buffer it described.
//
// Without the invalidation in ggml_gallocr_reserve_n_impl, the second
// alloc_graph of a same-shape graph found no reason to reserve again and placed
// tensors in a NULL vbuffer. Release builds define NDEBUG, so every check here
// is explicit and always runs.

#include "dummy-sched-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <cstdio>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

struct test_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    graph = nullptr;
    ggml_tensor *    out   = nullptr;
};

// n inputs of `bytes` each, summed by a chain of adds; the last is the output.
static test_graph make_graph(int n_inputs, size_t bytes) {
    ggml_init_params params{};
    params.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
    params.no_alloc = true;

    test_graph g;
    g.ctx.reset(ggml_init(params));
    g.graph = ggml_new_graph_custom(g.ctx.get(), 64, false);

    ggml_tensor * acc = nullptr;
    for (int i = 0; i < n_inputs; i++) {
        ggml_tensor * x = ggml_new_tensor_1d(g.ctx.get(), GGML_TYPE_F32, (int64_t) (bytes / 4));
        ggml_set_input(x);
        acc = acc ? ggml_add(g.ctx.get(), acc, x) : x;
    }
    g.out = ggml_add(g.ctx.get(), acc, acc);
    ggml_set_output(g.out);
    ggml_build_forward_expand(g.graph, g.out);
    return g;
}

static bool all_allocated(const test_graph & g) {
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); i++) {
        if (ggml_graph_node(g.graph, i)->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

// The registry of live backend buffers must agree with what the buffer type handed out: a
// refused reserve must not leave a half-registered or leaked buffer behind.
static bool live_matches(size_t baseline, const dummy_sched_backend & be) {
    return ggml_backend_test_live_buffer_count() == baseline + be.buffers.size();
}

int main() {
    const size_t live_baseline = ggml_backend_test_live_buffer_count();
    auto         be            = dummy_sched_backend::make(/*max_buffer_size=*/64);

    ggml_backend_t             backends[1] = { &be->backend };
    ggml_backend_buffer_type_t bufts[1]    = { &be->buffer_type };
    ggml_backend_sched_t       sched       = ggml_backend_sched_new(backends, bufts, 1, 64, false, false);
    CHECK(sched != nullptr, "setup: the scheduler must be created");

    // 1. A graph that fits allocates.
    {
        test_graph small = make_graph(2, 16);
        CHECK(ggml_backend_sched_alloc_graph(sched, small.graph),
              "case 1: a graph the buffer type accepts must allocate");
        CHECK(all_allocated(small), "case 1: every node must have a buffer");
        CHECK(!be->buffers.empty() && live_matches(live_baseline, *be),
              "case 1: the live buffer registry must hold exactly the buffers handed out");
        ggml_backend_sched_reset(sched);
    }

    // 2. A larger graph needs a reallocation, which the buffer type refuses.
    be->refuse_alloc = true;
    {
        test_graph big = make_graph(4, 24);
        CHECK(!ggml_backend_sched_alloc_graph(sched, big.graph), "case 2: a refused reallocation must return false");
        CHECK(be->n_refused >= 1, "case 2: the buffer type must have been asked and refused");
        CHECK(live_matches(live_baseline, *be), "case 2: a refused reallocation must not leak a live buffer");

        // 3. The same graph again, and a graph of the same shape: still false, and no crash.
        const int refused_before = be->n_refused;
        CHECK(!ggml_backend_sched_alloc_graph(sched, big.graph), "case 3: the same graph again must return false");
        test_graph again = make_graph(4, 24);
        CHECK(!ggml_backend_sched_alloc_graph(sched, again.graph), "case 3: a same-shape graph must return false");
        CHECK(be->n_refused > refused_before, "case 3: each retry must reserve again, not reuse a dead layout");
    }

    // 4. A refused reserve (the measure path) is the same: false, then false again.
    {
        test_graph measure = make_graph(5, 24);
        CHECK(!ggml_backend_sched_reserve(sched, measure.graph), "case 4: a refused reserve must return false");
        test_graph same = make_graph(5, 24);
        CHECK(!ggml_backend_sched_alloc_graph(sched, same.graph),
              "case 4: the alloc after a refused reserve must return false");
    }

    // 5. Once the buffer type accepts again, the same shapes allocate.
    be->refuse_alloc = false;
    {
        test_graph big = make_graph(4, 24);
        CHECK(ggml_backend_sched_alloc_graph(sched, big.graph),
              "case 5: the graph must allocate once the buffer type accepts");
        CHECK(all_allocated(big), "case 5: every node must have a buffer");
        ggml_backend_sched_reset(sched);
    }

    ggml_backend_sched_free(sched);
    CHECK(be->buffers.empty(), "teardown: no buffer may outlive the scheduler");
    CHECK(ggml_backend_test_live_buffer_count() == live_baseline,
          "teardown: the live buffer registry must be back at baseline");

    std::printf("PASS\n");
    return 0;
}
