// Host-only gate for the pure helpers in src/llama-measure-plan.h.
//
// llama_measure_graph_set() lists the graphs one planned reserve measures
// (zhcn design 2.3). The count formula (3 + S) * E * N is the design's own
// (Mistral -np 1: 6; GS1 -np 4 --no-kv-unified: 14), so each case checks the
// count against the formula and then the shape of the individual graphs:
//   - the pp and tg graphs are the allocating reserve's, verbatim;
//   - the s-stream graphs carry s * floor(n_ubatch / s) tokens, n_seqs = s,
//     n_streams = s and min(tokens, n_outputs_max) outputs, so the mask and the
//     K/V views agree on s;
//   - the nextn variants are off plus {on, on-masked} x each layer offset.
//
// llama_measure_chunk_plan() turns the chunk peaks of the measured graphs into
// slot caps. Below the highest chunk a cap is MAX(chunk size, largest peak);
// the last chunk is its high-water mark; a layout reaching the unbounded final
// chunk has no plan.
//
// No device, no model, no allocation.

#include "../src/llama-measure-plan.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static void expect(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        g_failures++;
    } else {
        std::printf("ok   %s\n", what);
    }
}

static llama_measure_set_params params(uint32_t n_tokens, uint32_t n_seq_max, uint32_t n_outputs_max, bool kv_unified,
                                       uint32_t n_layer_nextn = 0) {
    llama_measure_set_params p;
    p.n_tokens      = n_tokens;
    p.n_seq_max     = n_seq_max;
    p.n_outputs_max = n_outputs_max;
    p.kv_unified    = kv_unified;
    p.n_layer_nextn = n_layer_nextn;
    return p;
}

static size_t count_kind(const std::vector<llama_measure_graph> & g, llama_measure_kind k) {
    size_t n = 0;
    for (const auto & x : g) {
        n += x.kind == k;
    }
    return n;
}

static void test_counts() {
    // Mistral -np 1: pp, tg, pp again, for embeddings off and on.
    auto g = llama_measure_graph_set(params(512, 1, 2048, true));
    expect(g.size() == 6, "np 1 unified: (3 + 0) * 2 * 1 = 6 graphs");
    expect(count_kind(g, LLAMA_MEASURE_KIND_STREAM) == 0, "np 1 unified: no s-stream graph");

    // GS1: -np 4 --no-kv-unified.
    g = llama_measure_graph_set(params(512, 4, 2048, false));
    expect(g.size() == 14, "np 4 split: (3 + 4) * 2 * 1 = 14 graphs");
    expect(count_kind(g, LLAMA_MEASURE_KIND_STREAM) == 8, "np 4 split: 4 s-stream graphs per embeddings value");

    // Several sequences over a unified cache stay one stream.
    g = llama_measure_graph_set(params(512, 4, 2048, true));
    expect(g.size() == 6, "np 4 unified: no s-stream graph");

    // S is capped by the ubatch.
    g = llama_measure_graph_set(params(2, 8, 2048, false));
    expect(count_kind(g, LLAMA_MEASURE_KIND_STREAM) == 4, "n_ubatch 2 caps the streams at 2 (2 per embeddings value)");

    // Nextn: N = 1 + 2 * layers.
    g = llama_measure_graph_set(params(512, 1, 2048, true, 3));
    expect(g.size() == (3 + 0) * 2 * 7, "3 nextn layers: 3 * 2 * 7 = 42 graphs");
    g = llama_measure_graph_set(params(512, 4, 2048, false, 2));
    expect(g.size() == (3 + 4) * 2 * 5, "np 4 split, 2 nextn layers: 7 * 2 * 5 = 70 graphs");
}

static void test_shapes() {
    // pp and tg are the reserve's own graphs: n_outputs_pp = min(tokens, outputs_max).
    auto g = llama_measure_graph_set(params(512, 2, 100, true));
    const auto & pp = g[0];
    const auto & tg = g[1];
    const auto & again = g[2];
    expect(pp.kind == LLAMA_MEASURE_KIND_PP && pp.n_tokens == 512 && pp.n_seqs == 2 && pp.n_outputs == 100,
           "pp: n_tokens, n_seq_max sequences, min(tokens, n_outputs_max) outputs");
    expect(tg.kind == LLAMA_MEASURE_KIND_TG && tg.n_tokens == 2 && tg.n_seqs == 2 && tg.n_outputs == 2,
           "tg: n_seq_max tokens, sequences and outputs");
    expect(again.kind == LLAMA_MEASURE_KIND_PP_AGAIN && again.n_tokens == 512 && again.n_seqs == 2 &&
               again.n_outputs == 100,
           "pp again repeats pp");
    expect(pp.n_streams == 0 && tg.n_streams == 0 && again.n_streams == 0, "the reserve graphs span every stream");

    // The archs whose pp compute grows with n_seq_tokens^2 close on one sequence.
    auto p = params(512, 4, 2048, true);
    p.pp_again_single_seq = true;
    g = llama_measure_graph_set(p);
    expect(g[0].n_seqs == 4 && g[2].n_seqs == 1 && g[2].n_tokens == 512, "pp again on one sequence for the quadratic-mask archs");

    // The embeddings value splits the set in two halves of equal size.
    g = llama_measure_graph_set(params(512, 1, 2048, true));
    expect(!g[0].embeddings && !g[1].embeddings && !g[2].embeddings && g[3].embeddings && g[4].embeddings && g[5].embeddings,
           "embeddings off first, then on");

    // Warmup marks every graph while it is on.
    p = params(512, 1, 2048, true);
    p.warmup = true;
    g = llama_measure_graph_set(p);
    bool all = true;
    for (const auto & x : g) {
        all = all && x.warmup;
    }
    expect(all && g.size() == 6, "warmup on: every graph is a warmup graph, none added");
}

static void test_streams() {
    // 512 tokens over s streams: s * floor(512 / s), n_seqs = n_streams = s.
    auto g = llama_measure_graph_set(params(512, 5, 2048, false));
    std::vector<llama_measure_graph> st;
    for (const auto & x : g) {
        if (x.kind == LLAMA_MEASURE_KIND_STREAM && !x.embeddings) {
            st.push_back(x);
        }
    }
    expect(st.size() == 5, "five s-stream graphs for np 5");
    bool ok = true;
    for (size_t i = 0; i < st.size(); ++i) {
        const uint32_t s = (uint32_t) i + 1;
        ok = ok && st[i].n_streams == s && st[i].n_seqs == s && st[i].n_tokens == s * (512 / s) &&
             st[i].n_outputs == st[i].n_tokens;
    }
    expect(ok, "s-stream shape: tokens s * floor(512 / s), n_seqs = n_streams = s, outputs = tokens");
    expect(st[2].n_tokens == 510 && st[4].n_tokens == 510, "s = 3 and s = 5 measure 510 tokens, never the rounded-up 513 or 515");

    // The output count is capped like the pp graph's.
    g = llama_measure_graph_set(params(512, 4, 100, false));
    ok = true;
    for (const auto & x : g) {
        if (x.kind == LLAMA_MEASURE_KIND_STREAM) {
            ok = ok && x.n_outputs == 100;
        }
    }
    expect(ok, "s-stream outputs are min(tokens, n_outputs_max)");
}

static void test_nextn() {
    auto g = llama_measure_graph_set(params(512, 1, 2048, true, 2));
    // embeddings off: variants 0..4, three graphs each.
    std::vector<llama_measure_graph> pp;
    for (const auto & x : g) {
        if (x.kind == LLAMA_MEASURE_KIND_PP && !x.embeddings) {
            pp.push_back(x);
        }
    }
    expect(pp.size() == 5, "off + {on, on-masked} x 2 offsets = 5 variants");
    expect(!pp[0].nextn && pp[0].nextn_offset == 0, "variant 0 is nextn off");
    expect(pp[1].nextn && !pp[1].nextn_masked && pp[1].nextn_offset == 0, "variant 1: on, offset 0");
    expect(pp[2].nextn && pp[2].nextn_masked && pp[2].nextn_offset == 0, "variant 2: on-masked, offset 0");
    expect(pp[3].nextn && !pp[3].nextn_masked && pp[3].nextn_offset == 1, "variant 3: on, offset 1");
    expect(pp[4].nextn && pp[4].nextn_masked && pp[4].nextn_offset == 1, "variant 4: on-masked, offset 1");

    // Without nextn layers the fields are inert and never enumerated.
    g = llama_measure_graph_set(params(512, 1, 2048, true, 0));
    bool any = false;
    for (const auto & x : g) {
        any = any || x.nextn;
    }
    expect(!any, "no nextn layers: no nextn variant");
}

static std::vector<std::vector<size_t>> peaks(std::initializer_list<std::vector<size_t>> l) {
    return std::vector<std::vector<size_t>>(l);
}

static void test_chunk_plan() {
    std::vector<size_t> cap;
    std::string         reason;

    // One chunk: the plan is the largest measured peak, nothing over-planned.
    bool ok = llama_measure_chunk_plan(peaks({ { 700 }, { 300 }, { 900 } }), 2000, 16, cap, reason);
    expect(ok && cap.size() == 1 && cap[0] == 900, "single chunk: the high-water mark, not the chunk size");

    // Several chunks: below the last, MAX(chunk size, largest peak); the last is the high-water mark.
    ok = llama_measure_chunk_plan(peaks({ { 1000, 1000, 400 }, { 1000, 600, 0 } }), 1000, 16, cap, reason);
    expect(ok && cap.size() == 3 && cap[0] == 1000 && cap[1] == 1000 && cap[2] == 400,
           "three chunks: full chunk size below the last, the peak in the last");

    // An under-filled chunk below the last: a smaller decode ubatch can fill it to its capacity.
    ok = llama_measure_chunk_plan(peaks({ { 700, 300 }, { 650, 100 } }), 1000, 16, cap, reason);
    expect(ok && cap.size() == 2 && cap[0] == 1000 && cap[1] == 300, "an under-filled chunk 0 plans its full capacity");

    // An oversize chunk below the last keeps its own peak as its capacity.
    ok = llama_measure_chunk_plan(peaks({ { 3000, 500 }, { 800, 100 } }), 1000, 16, cap, reason);
    expect(ok && cap.size() == 2 && cap[0] == 3000 && cap[1] == 500, "an oversize chunk 0 is its own peak");

    // A graph with fewer chunks contributes only to the indices it has.
    ok = llama_measure_chunk_plan(peaks({ { 100 }, { 1000, 250 } }), 1000, 16, cap, reason);
    expect(ok && cap.size() == 2 && cap[0] == 1000 && cap[1] == 250, "a shorter layout leaves the later chunks to the longer one");

    // The unbounded final gallocr chunk is a named refusal.
    ok = llama_measure_chunk_plan(peaks({ { 1, 2, 3, 4 } }), 1000, 4, cap, reason);
    expect(!ok && cap.empty() && reason == "graph needs the unbounded final gallocr chunk",
           "reaching chunk index max_chunks - 1 refuses");
    ok = llama_measure_chunk_plan(peaks({ { 1, 2, 3 } }), 1000, 4, cap, reason);
    expect(ok && cap.size() == 3, "one chunk below the unbounded index is fine");

    // Several chunks with no chunk size cannot be planned.
    ok = llama_measure_chunk_plan(peaks({ { 1, 2 } }), SIZE_MAX, 16, cap, reason);
    expect(!ok && !reason.empty(), "several chunks with an unbounded chunk size refuse");
    ok = llama_measure_chunk_plan(peaks({ { 5 } }), SIZE_MAX, 16, cap, reason);
    expect(ok && cap.size() == 1 && cap[0] == 5, "one chunk with an unbounded chunk size is its peak");

    // No graph, no chunk: an empty plan.
    ok = llama_measure_chunk_plan(peaks({}), 1000, 16, cap, reason);
    expect(ok && cap.empty(), "no measured chunk: an empty plan");
}

int main() {
    test_counts();
    test_shapes();
    test_streams();
    test_nextn();
    test_chunk_plan();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d case(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all cases passed\n");
    return 0;
}
