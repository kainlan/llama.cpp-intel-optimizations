// The scheduler reports the chunk layout of its last reserve, so a planner can
// size one slot per chunk index. The capacity of chunk c of a measured set of
// graphs is
//   cap[c] = MAX(M, max over graphs of peak[c])   for c below the last chunk
//   cap[L] = max over graphs of peak[L]           for the last chunk L
// where M is the allocator's max chunk size. Every buffer a later reserve asks
// the buffer type for, at chunk index c, must fit cap[c]. Release builds define
// NDEBUG, so every check here is explicit and always runs.

#include "dummy-sched-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static const int MAX_CHUNKS = 16;

struct test_graph {
    ggml_context_ptr           ctx;
    ggml_cgraph *              graph = nullptr;
    std::vector<ggml_tensor *> inputs;
};

// Inputs of the given byte sizes are allocated first, in order, each in the
// first chunk with room. An op over views of them makes the graph a graph.
static test_graph make_graph(const std::vector<size_t> & input_bytes) {
    ggml_init_params params{};
    params.mem_size = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
    params.no_alloc = true;

    test_graph g;
    g.ctx.reset(ggml_init(params));
    g.graph = ggml_new_graph_custom(g.ctx.get(), 64, false);

    ggml_tensor * acc = nullptr;
    for (size_t bytes : input_bytes) {
        ggml_tensor * x = ggml_new_tensor_1d(g.ctx.get(), GGML_TYPE_F32, (int64_t) (bytes / 4));
        ggml_set_input(x);
        g.inputs.push_back(x);
        if (!acc) {
            acc = x;
        } else {
            const int64_t n = std::min(acc->ne[0], x->ne[0]);
            acc = ggml_add(g.ctx.get(), ggml_view_1d(g.ctx.get(), acc, n, 0), ggml_view_1d(g.ctx.get(), x, n, 0));
        }
    }
    ggml_set_output(acc);
    ggml_build_forward_expand(g.graph, acc);
    return g;
}

struct layout {
    std::vector<size_t> peaks;
    size_t              max_chunk_size = 0;
};

static bool read_layout(ggml_backend_sched_t sched, ggml_backend_t backend, layout * out) {
    size_t    peaks[MAX_CHUNKS] = {};
    size_t    m                 = 0;
    const int n                 = ggml_backend_sched_get_reserved_chunk_peaks(sched, backend, peaks, MAX_CHUNKS, &m);
    if (n < 0 || n > MAX_CHUNKS) {
        return false;
    }
    out->peaks.assign(peaks, peaks + n);
    out->max_chunk_size = m;
    return true;
}

// cap[c] over a set of measured layouts of one buffer type.
static std::vector<size_t> caps_of(const std::vector<layout> & layouts) {
    size_t n_chunks = 0;
    size_t m        = 0;
    for (const layout & l : layouts) {
        n_chunks = std::max(n_chunks, l.peaks.size());
        m        = l.max_chunk_size;
    }
    std::vector<size_t> cap(n_chunks, 0);
    for (const layout & l : layouts) {
        for (size_t c = 0; c < l.peaks.size(); c++) {
            cap[c] = std::max(cap[c], l.peaks[c]);
        }
    }
    for (size_t c = 0; c + 1 < n_chunks; c++) {
        cap[c] = std::max(cap[c], m);
    }
    return cap;
}

static bool requests_fit(const std::vector<size_t> & requests, const std::vector<size_t> & cap, int * bad_index) {
    // a reserve asks for chunk 0..k-1 in order; a request run restarts at chunk 0
    // whenever a new vbuffer is allocated, so match the runs by length of the layout
    // that produced them: the caller passes the requests of ONE reserve.
    for (size_t i = 0; i < requests.size(); i++) {
        if (i >= cap.size() || requests[i] > cap[i]) {
            *bad_index = (int) i;
            return false;
        }
    }
    return true;
}

int main() {
    // ---- one buffer type, max chunk size 64: pp, then a tg that grows chunk 1 only, then pp ----
    {
        auto be = dummy_sched_backend::make(/*max_buffer_size=*/64);

        ggml_backend_t             backends[1] = { &be->backend };
        ggml_backend_buffer_type_t bufts[1]    = { &be->buffer_type };
        ggml_backend_sched_t       sched       = ggml_backend_sched_new(backends, bufts, 1, 64, false, false);
        CHECK(sched != nullptr, "setup: the scheduler must be created");

        const std::vector<size_t> pp_shape = { 8, 8, 8, 40 };
        const std::vector<size_t> tg_shape = { 8, 8, 16, 40 };

        // Measure: the layout of each graph, read straight after its reserve_size.
        std::vector<layout> measured;
        for (const std::vector<size_t> * shape : { &pp_shape, &tg_shape, &pp_shape }) {
            test_graph g = make_graph(*shape);
            size_t     sizes[1];
            ggml_backend_sched_reserve_size(sched, g.graph, sizes);
            layout l;
            CHECK(read_layout(sched, &be->backend, &l), "case 1: the layout must be readable");
            CHECK(l.max_chunk_size == 64, "case 1: the query must report the buffer type's max chunk size");
            size_t sum = 0;
            for (size_t p : l.peaks) {
                sum += p;
            }
            CHECK(sum == sizes[0], "case 1: the peaks must sum to the size reserve_size reports");
            measured.push_back(l);
            ggml_backend_sched_reset(sched);
        }

        // The scenario is the one the cap rule exists for: tg grows chunk 1 only.
        CHECK(measured[0].peaks.size() == 2 && measured[1].peaks.size() == 2, "case 1: both graphs use two chunks");
        CHECK(measured[1].peaks[0] < measured[0].peaks[0], "case 1: tg must leave chunk 0 under-filled");
        CHECK(measured[1].peaks[1] > measured[0].peaks[1], "case 1: tg must grow chunk 1");

        const std::vector<size_t> cap = caps_of(measured);
        CHECK(cap.size() == 2 && cap[0] == 64 && cap[1] == measured[1].peaks[1],
              "case 1: cap[0] is the max chunk size, cap[1] is the last chunk's high-water mark");

        // Control: a per-chunk cap derived from the totals alone (total - M for the
        // spill chunk) would have refused tg's chunk-1 request. The query is needed.
        {
            size_t naive_cap1 = 0;
            for (const layout & l : measured) {
                size_t total = 0;
                for (size_t p : l.peaks) {
                    total += p;
                }
                naive_cap1 = std::max(naive_cap1, total > l.max_chunk_size ? total - l.max_chunk_size : 0);
            }
            CHECK(naive_cap1 < measured[1].peaks[1], "case 1 control: a total-derived cap must be too small here");
        }

        // Allocate pp, tg, pp for real: every request at chunk index c fits cap[c].
        const std::vector<size_t> * order[3]          = { &pp_shape, &tg_shape, &pp_shape };
        bool                        saw_chunk1_growth = false;
        for (const std::vector<size_t> * shape : order) {
            test_graph g = make_graph(*shape);
            be->requests.clear();
            CHECK(ggml_backend_sched_reserve(sched, g.graph), "case 1: the reserve must succeed");
            int bad = -1;
            CHECK(requests_fit(be->requests, cap, &bad), "case 1: a buffer request exceeded its chunk's planned cap");
            if (be->requests.size() == 2 && be->requests[1] > measured[0].peaks[1]) {
                saw_chunk1_growth = true;
            }
        }
        CHECK(saw_chunk1_growth, "case 1: the logging buffer type must have seen the grown chunk 1");

        // An under-filled chunk 0 is still planned at the max chunk size: measured on tg alone
        // (chunk 0 peak below M), the cap is M, so a later pp that fills chunk 0 is not refused.
        {
            const std::vector<size_t> cap_tg_only = caps_of({ measured[1] });
            CHECK(measured[1].peaks[0] < 64, "case 1 control: tg's chunk 0 peak is below M");
            CHECK(cap_tg_only.size() == 2 && cap_tg_only[0] == 64, "case 1: an under-filled chunk 0 is planned at M");
        }

        // Decode-sized graphs: every smaller shape stays within cap too.
        for (size_t big = 8; big <= 40; big += 8) {
            test_graph g = make_graph({ big, 8, 8 });
            be->requests.clear();
            CHECK(ggml_backend_sched_reserve(sched, g.graph), "case 1: a smaller reserve must succeed");
            int bad = -1;
            CHECK(requests_fit(be->requests, cap, &bad), "case 1: a smaller graph's buffer request exceeded its cap");
        }

        ggml_backend_sched_free(sched);
        CHECK(be->buffers.empty(), "case 1 teardown: no buffer may outlive the scheduler");
    }

    // ---- an oversize chunk below the last: its peak is its capacity ----
    {
        auto be = dummy_sched_backend::make(/*max_buffer_size=*/64);

        ggml_backend_t             backends[1] = { &be->backend };
        ggml_backend_buffer_type_t bufts[1]    = { &be->buffer_type };
        ggml_backend_sched_t       sched       = ggml_backend_sched_new(backends, bufts, 1, 64, false, false);
        CHECK(sched != nullptr, "setup: the scheduler must be created");

        test_graph g = make_graph({ 8, 96, 96 });
        size_t     sizes[1];
        ggml_backend_sched_reserve_size(sched, g.graph, sizes);
        layout l;
        CHECK(read_layout(sched, &be->backend, &l), "case 2: the layout must be readable");
        CHECK(l.peaks.size() >= 3, "case 2: two oversize tensors and a small one need three chunks");

        const std::vector<size_t> cap = caps_of({ l });
        for (size_t c = 0; c + 1 < l.peaks.size(); c++) {
            if (l.peaks[c] > l.max_chunk_size) {
                CHECK(cap[c] == l.peaks[c], "case 2: an oversize chunk's cap is its peak");
            } else {
                CHECK(cap[c] == l.max_chunk_size, "case 2: a non-oversize chunk below the last has the max chunk size");
            }
        }

        be->requests.clear();
        ggml_backend_sched_reset(sched);
        CHECK(ggml_backend_sched_reserve(sched, g.graph), "case 2: the reserve must succeed");
        int bad = -1;
        CHECK(requests_fit(be->requests, cap, &bad), "case 2: a buffer request exceeded its chunk's cap");

        // The derived capacity: an oversize chunk holds one tensor at offset 0.
        // (A reserve places nothing, so allocate the graph to read the offsets.)
        CHECK(ggml_backend_sched_alloc_graph(sched, g.graph), "case 2: the graph must allocate");
        for (ggml_tensor * x : g.inputs) {
            if (ggml_nbytes(x) > l.max_chunk_size) {
                CHECK(
                    x->buffer != nullptr && (uintptr_t) x->data == (uintptr_t) ggml_backend_buffer_get_base(x->buffer),
                    "case 2: an oversize tensor must sit at offset 0 of its own chunk");
            }
        }

        ggml_backend_sched_free(sched);
    }

    // ---- two buffer types with different max chunk sizes ----
    {
        auto a = dummy_sched_backend::make(/*max_buffer_size=*/64);
        auto b = dummy_sched_backend::make(/*max_buffer_size=*/32);

        ggml_backend_t             backends[2] = { &a->backend, &b->backend };
        ggml_backend_buffer_type_t bufts[2]    = { &a->buffer_type, &b->buffer_type };
        ggml_backend_sched_t       sched       = ggml_backend_sched_new(backends, bufts, 2, 64, false, false);
        CHECK(sched != nullptr, "setup: the two-backend scheduler must be created");

        // Two independent chains: the first on backend a, the second on backend b.
        auto build = [&](size_t first, size_t second, test_graph * out) {
            ggml_init_params params{};
            params.mem_size = 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
            params.no_alloc = true;
            out->ctx.reset(ggml_init(params));
            ggml_context * ctx = out->ctx.get();
            out->graph         = ggml_new_graph_custom(ctx, 64, false);

            ggml_tensor * x0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (first / 4));
            ggml_tensor * x1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (first / 4));
            ggml_tensor * y0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (second / 4));
            ggml_tensor * y1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) (second / 4));
            for (ggml_tensor * t : { x0, x1, y0, y1 }) {
                ggml_set_input(t);
            }
            ggml_tensor * xo = ggml_add(ctx, x0, x1);
            ggml_tensor * yo = ggml_add(ctx, y0, y1);
            ggml_set_output(xo);
            ggml_set_output(yo);
            ggml_build_forward_expand(out->graph, xo);
            ggml_build_forward_expand(out->graph, yo);
            ggml_backend_sched_set_tensor_backend(sched, x0, &a->backend);
            ggml_backend_sched_set_tensor_backend(sched, x1, &a->backend);
            ggml_backend_sched_set_tensor_backend(sched, xo, &a->backend);
            ggml_backend_sched_set_tensor_backend(sched, y0, &b->backend);
            ggml_backend_sched_set_tensor_backend(sched, y1, &b->backend);
            ggml_backend_sched_set_tensor_backend(sched, yo, &b->backend);
        };

        std::vector<layout> la, lb;
        const size_t        shapes[3][2] = {
            { 48, 24 },
            { 24, 40 },
            { 48, 24 }
        };
        for (const auto & s : shapes) {
            test_graph g;
            build(s[0], s[1], &g);
            CHECK(ggml_backend_sched_reserve(sched, g.graph), "case 3: the reserve must succeed");
            layout x, y;
            CHECK(read_layout(sched, &a->backend, &x) && read_layout(sched, &b->backend, &y),
                  "case 3: both layouts must be readable");
            CHECK(x.max_chunk_size == 64 && y.max_chunk_size == 32,
                  "case 3: each buffer type reports its own max chunk size");
            la.push_back(x);
            lb.push_back(y);
        }
        const std::vector<size_t> cap_a = caps_of(la);
        const std::vector<size_t> cap_b = caps_of(lb);

        // Second pass: allocate again over the same sched and check each type's requests.
        for (const auto & s : shapes) {
            test_graph g;
            build(s[0], s[1], &g);
            a->requests.clear();
            b->requests.clear();
            CHECK(ggml_backend_sched_reserve(sched, g.graph), "case 3: the second-pass reserve must succeed");
            int bad = -1;
            CHECK(requests_fit(a->requests, cap_a, &bad), "case 3: backend a's request exceeded its chunk's cap");
            CHECK(requests_fit(b->requests, cap_b, &bad), "case 3: backend b's request exceeded its chunk's cap");
        }

        ggml_backend_sched_free(sched);
    }

    std::printf("PASS\n");
    return 0;
}
