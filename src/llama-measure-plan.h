#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Pure helpers behind the planned reserve's MEASURE pass
// (llama_context::sched_reserve_impl, mode MEASURE). They take plain integers
// and touch no context, scheduler or backend state, so
// tests/test-measure-plan.cpp executes them on the host.
//
//   llama_measure_graph_set()  the graphs a context can reach at allocation,
//                              in the order the pass measures them;
//   llama_measure_chunk_plan() the per-chunk slot caps from the chunk peaks
//                              the measured graphs left in the scheduler.

enum llama_measure_kind : uint32_t {
    LLAMA_MEASURE_KIND_PP,        // the worst-case prompt-processing graph
    LLAMA_MEASURE_KIND_TG,        // the token-generation graph
    LLAMA_MEASURE_KIND_PP_AGAIN,  // the reserve's closing pp graph (it may differ for some archs)
    LLAMA_MEASURE_KIND_STREAM,    // a decode ubatch over s streams of a split KV cache
};

// One graph to reserve. `n_streams` is the stream count of the reserve memory
// context (llama_memory_i::init_reserve); 0 means every stream of the memory,
// which is what init_full() builds.
struct llama_measure_graph {
    llama_measure_kind kind          = LLAMA_MEASURE_KIND_PP;
    uint32_t           n_tokens      = 0;
    uint32_t           n_seqs        = 0;
    uint32_t           n_streams     = 0;
    uint32_t           n_outputs     = 0;
    bool               embeddings    = false;
    bool               nextn         = false;
    bool               nextn_masked  = false;
    int32_t            nextn_offset  = 0;
    bool               warmup        = false;
};

struct llama_measure_set_params {
    uint32_t n_tokens      = 0;      // min(n_ctx, n_ubatch): the pp graph's token count
    uint32_t n_seq_max     = 1;
    uint32_t n_outputs_max = 0;
    bool     kv_unified    = true;
    uint32_t n_layer_nextn = 0;      // hparams.n_layer_nextn
    bool     warmup        = false;  // the context's warmup flag now: while on, every graph is a warmup graph
    // The reserve's closing pp graph is built with one sequence for the archs
    // whose pp compute grows with n_seq_tokens^2 (KIMI_LINEAR, MINIMAX_01).
    bool     pp_again_single_seq = false;
};

// The number of nextn variants: off, plus {on, on-masked} for each of the
// n_layer_nextn layer offsets. Without nextn layers the fields are inert.
inline uint32_t llama_measure_nextn_variants(uint32_t n_layer_nextn) {
    return 1 + 2 * n_layer_nextn;
}

// The streams a decode ubatch can span: min(n_seq_max, n_ubatch) for a split
// KV cache with several sequences, 0 otherwise.
inline uint32_t llama_measure_stream_count(uint32_t n_seq_max, uint32_t n_tokens, bool kv_unified) {
    return (!kv_unified && n_seq_max > 1) ? std::min(n_seq_max, n_tokens) : 0;
}

// The graphs one reserve measures: for each embeddings value {off, on} and each
// nextn variant, the three graphs the allocating reserve builds (pp, tg, pp
// again), then one decode graph per stream count s in [1, S]. That is
// (3 + S) * 2 * N graphs, with N = llama_measure_nextn_variants(). A smaller
// decode ubatch is not enumerated: the chunk rule bounds it.
inline std::vector<llama_measure_graph> llama_measure_graph_set(const llama_measure_set_params & p) {
    std::vector<llama_measure_graph> out;

    const uint32_t n_outputs_pp = std::min(p.n_tokens, p.n_outputs_max);
    const uint32_t n_streams    = llama_measure_stream_count(p.n_seq_max, p.n_tokens, p.kv_unified);

    for (int emb = 0; emb < 2; ++emb) {
        for (uint32_t v = 0; v < llama_measure_nextn_variants(p.n_layer_nextn); ++v) {
            llama_measure_graph base;
            base.embeddings   = emb != 0;
            base.warmup       = p.warmup;
            if (v > 0) {
                base.nextn        = true;
                base.nextn_masked = ((v - 1) % 2) == 1;
                base.nextn_offset = (int32_t) ((v - 1) / 2);
            }

            llama_measure_graph pp = base;
            pp.kind      = LLAMA_MEASURE_KIND_PP;
            pp.n_tokens  = p.n_tokens;
            pp.n_seqs    = p.n_seq_max;
            pp.n_outputs = n_outputs_pp;
            out.push_back(pp);

            llama_measure_graph tg = base;
            tg.kind      = LLAMA_MEASURE_KIND_TG;
            tg.n_tokens  = p.n_seq_max;
            tg.n_seqs    = p.n_seq_max;
            tg.n_outputs = p.n_seq_max;
            out.push_back(tg);

            llama_measure_graph again = pp;
            again.kind = LLAMA_MEASURE_KIND_PP_AGAIN;
            if (p.pp_again_single_seq) {
                again.n_seqs = 1;
            }
            out.push_back(again);

            for (uint32_t s = 1; s <= n_streams; ++s) {
                llama_measure_graph st = base;
                st.kind      = LLAMA_MEASURE_KIND_STREAM;
                st.n_tokens  = s * (p.n_tokens / s);
                st.n_seqs    = s;
                st.n_streams = s;
                st.n_outputs = std::min(st.n_tokens, p.n_outputs_max);
                out.push_back(st);
            }
        }
    }

    return out;
}

// The slot caps of one compute buft from the chunk layouts the measured graphs
// left in the scheduler. `peaks[g][c]` is graph g's peak in chunk c (a graph
// with fewer chunks than c contributes nothing to c), and `max_chunk_size` is
// the chunk capacity gallocr was given for the buft, the same for every graph.
//
// gallocr fixes a chunk's capacity at creation as MAX(peak, max_chunk_size), and
// an oversize chunk's tensor sits at offset 0, so below the highest chunk index
// L every chunk's capacity is MAX(max_chunk_size, its largest peak): a smaller
// decode ubatch can fill an under-filled chunk up to that and never beyond. The
// last chunk is only ever filled to its high-water mark. A layout that reaches
// the gallocr chunk index `max_chunks - 1` is unbounded and has no cap.
//
// Returns false with `reason` set, and an empty `cap`, when no plan exists.
inline bool llama_measure_chunk_plan(const std::vector<std::vector<size_t>> & peaks,
                                     size_t                                   max_chunk_size,
                                     size_t                                   max_chunks,
                                     std::vector<size_t> &                    cap,
                                     std::string &                            reason) {
    cap.clear();
    reason.clear();

    size_t n_chunks = 0;
    for (const auto & g : peaks) {
        n_chunks = std::max(n_chunks, g.size());
    }
    if (n_chunks == 0) {
        return true;
    }
    if (n_chunks >= max_chunks) {
        reason = "graph needs the unbounded final gallocr chunk";
        return false;
    }
    if (n_chunks > 1 && max_chunk_size == SIZE_MAX) {
        reason = "several gallocr chunks with no chunk size";
        return false;
    }

    std::vector<size_t> hi(n_chunks, 0);
    for (const auto & g : peaks) {
        for (size_t c = 0; c < g.size(); ++c) {
            hi[c] = std::max(hi[c], g[c]);
        }
    }

    cap.resize(n_chunks);
    for (size_t c = 0; c + 1 < n_chunks; ++c) {
        cap[c] = std::max(max_chunk_size, hi[c]);
    }
    cap[n_chunks - 1] = hi[n_chunks - 1];
    return true;
}
