#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// Pure helpers behind the SYCL auto micro-batch trial
// (llama_context::sycl_select_auto_ubatch). They take plain integers and
// touch no context or backend state, so tests/test-auto-ubatch-ladder.cpp
// executes them on the host.

// The one exception a compute-buffer reserve throws when the rung's compute buffers did not fit: a fit verdict, the
// kind of loss that lets the auto-ubatch trial descend. Everything else a reserve throws (a lifecycle result such as
// BUSY or STALE_IDENTITY, a memory module that would not initialize) is no verdict on the rung and never lowers -ub.
struct llama_auto_ubatch_fit_refusal : public std::runtime_error {
    explicit llama_auto_ubatch_fit_refusal(const std::string & what) : std::runtime_error(what) {}
};

// The trial's candidate micro-batch sizes, ascending.
static const uint32_t llama_auto_ubatch_ladder[] = { 512, 1024, 2048, 4096 };
static const size_t   llama_auto_ubatch_ladder_size =
    sizeof(llama_auto_ubatch_ladder) / sizeof(llama_auto_ubatch_ladder[0]);

// The smallest n_ubatch the trial lowers itself to (llama.cpp-kpjw).
static const uint32_t llama_auto_ubatch_descent_floor = 64;

// The next rung of the trial's downward continuation: half of `from`, rounded down to a multiple of 64, or 0 when
// that is under llama_auto_ubatch_descent_floor (nothing below it) or `from` is 0. When the default rung, and so
// every rung above it, is refused, the trial lowers n_ubatch rather than refusing the context: a smaller -ub is not
// a smaller context (B50, Qwen3.6-27B auto: 512 spilled a 495 MB compute buffer outside the arena and left 107.7 MB
// against the 256 MB driver headroom; 256 spills half of that and fits).
inline uint32_t llama_auto_ubatch_next_lower(uint32_t from) {
    const uint32_t half = from / 2;
    return half >= llama_auto_ubatch_descent_floor ? half - half % 64 : 0;
}

// The downward continuation itself: `try_rung(c)` is asked about each rung below `fallback` in turn (largest
// first, rungs above `cap` skipped, not asked) and answers true for a rung that fits. Returns the first rung that
// fits, or 0 when none does. The trial runs this only after the ascending ladder found nothing at or above the
// default, so the answer is the largest rung under it, never a smaller one than needed.
template <typename F> inline uint32_t llama_auto_ubatch_descend(uint32_t fallback, uint32_t cap, F try_rung) {
    for (uint32_t c = llama_auto_ubatch_next_lower(fallback); c != 0; c = llama_auto_ubatch_next_lower(c)) {
        if (c > cap) {
            continue;
        }
        if (try_rung(c)) {
            return c;
        }
    }
    return 0;
}

// The -ub a refusal names (llama.cpp-kpjw, kpjw-g7). `largest_fit` is the answer of the one hold-spill fit function (the
// largest -ub it accepts, 0 when none is known to fit); `lowest_refused` is the smallest rung this start already asked
// and lost (0: none yet). The advice is never a rung that was refused: an answer under `lowest_refused` stands, and
// one at or above it (the fit function accepts a rung that lost for another reason) is capped to the largest power of
// two strictly under it. Under the descent floor there is nothing to name, so the answer is 0.
inline uint32_t llama_auto_ubatch_advice(uint32_t largest_fit, uint32_t lowest_refused) {
    if (largest_fit == 0) {
        return 0;
    }
    if (lowest_refused == 0 || largest_fit < lowest_refused) {
        return largest_fit;
    }
    uint32_t p = 1;
    while (p <= lowest_refused / 2 && p * 2 < lowest_refused) {
        p *= 2;
    }
    return lowest_refused > 1 && p >= llama_auto_ubatch_descent_floor ? p : 0;
}

// True iff some rung of `ladder` lies in [ubatch_floor, ubatch_cap]. The
// trial's ladder loop skips every rung below `ubatch_floor` (the caller's own
// n_ubatch, which the trial never shrinks) and stops at the first rung above
// `ubatch_cap`, so without such a rung the loop cannot try a single candidate.
// Both bounds matter: `ubatch_floor` can sit between two rungs with
// `ubatch_cap` below the next one, and it can exceed `ubatch_cap` outright;
// neither case needs it to be above the ladder's largest rung.
inline bool llama_auto_ubatch_ladder_has_candidate(const uint32_t * ladder,
                                                   size_t           n_ladder,
                                                   uint32_t         ubatch_floor,
                                                   uint32_t         ubatch_cap) {
    for (size_t i = 0; i < n_ladder; ++i) {
        if (ladder[i] >= ubatch_floor && ladder[i] <= ubatch_cap) {
            return true;
        }
    }
    return false;
}

// True iff the trial's settle step must republish last_good on every device.
// A candidate publish that took effect (published_any), or that threw and so
// may have landed on some devices before one refused (publish_dirty), leaves
// device plans that need not describe last_good. Otherwise the constructor's
// own publish of fallback_ubatch still stands. At the trial's call site a
// changed n_ubatch already implies one of the two flags, since n_ubatch is
// written only just before a publish attempt; the n_ubatch != fallback_ubatch
// term is a defensive backstop in case a future caller changes n_ubatch without
// publishing.
inline bool llama_auto_ubatch_settle_needs_publish(bool     published_any,
                                                   bool     publish_dirty,
                                                   uint32_t n_ubatch,
                                                   uint32_t fallback_ubatch) {
    return published_any || publish_dirty || n_ubatch != fallback_ubatch;
}

// True iff the trial runs at all: the caller did not pin -ub (n_ubatch_auto),
// the model is causal (a non-causal model's n_ubatch == n_batch semantics are
// never shrunk), the context has a SYCL backend, and GGML_SYCL_AUTO_UBATCH
// allows it. The constructor passes the four evaluated results, so the trial
// decision and anything that plans for the trial's rungs cannot disagree.
inline bool llama_auto_ubatch_trial_runs(bool n_ubatch_auto,
                                         bool causal_attn,
                                         bool has_sycl_backend,
                                         bool auto_ubatch_enabled) {
    return n_ubatch_auto && causal_attn && has_sycl_backend && auto_ubatch_enabled;
}

// The trial's cap: min(n_batch, n_ctx), narrowed to the GPU MoE routing
// ceiling for a MoE model whenever that ceiling does not exceed it. A ceiling
// equal to the cap still binds, so the stop reason names the ceiling.
// `moe_cap_available` is false when the backend does not export the ceiling
// (an older SYCL DSO); the caller then passes any moe_cap and no narrowing
// happens. *moe_bound is set to whether the ceiling is the binding cap.
inline uint32_t llama_auto_ubatch_cap(uint32_t n_batch,
                                      uint32_t n_ctx,
                                      uint32_t n_expert,
                                      uint32_t moe_cap,
                                      bool     moe_cap_available,
                                      bool *   moe_bound) {
    uint32_t cap = n_batch < n_ctx ? n_batch : n_ctx;
    *moe_bound   = false;
    if (n_expert > 0 && moe_cap_available && moe_cap <= cap) {
        cap        = moe_cap;
        *moe_bound = true;
    }
    return cap;
}

// True iff a tuning-cache value may win: not below the first rung, not above
// the cap, and not below the caller's own n_ubatch (the trial never shrinks it).
inline bool llama_auto_ubatch_cached_valid(const uint32_t * ladder,
                                           size_t           n_ladder,
                                           uint32_t         cached_ubatch,
                                           uint32_t         fallback_ubatch,
                                           uint32_t         cap) {
    return n_ladder > 0 && cached_ubatch >= ladder[0] && cached_ubatch <= cap && cached_ubatch >= fallback_ubatch;
}

// Room for the largest rung set: every rung, plus the fallback, plus a cached
// value that is not itself a rung.
static const size_t llama_auto_ubatch_rung_set_capacity = llama_auto_ubatch_ladder_size + 2;

// Inserts v into the ascending array out[0..n) unless already present, and
// returns the new length. A full array is left unchanged.
inline size_t llama_auto_ubatch_set_insert(uint32_t * out, size_t n, size_t max_out, uint32_t v) {
    size_t pos = 0;
    while (pos < n && out[pos] < v) {
        ++pos;
    }
    if ((pos < n && out[pos] == v) || n >= max_out) {
        return n;
    }
    for (size_t k = n; k > pos; --k) {
        out[k] = out[k - 1];
    }
    out[pos] = v;
    return n + 1;
}

// The micro-batch sizes a trial may settle on or revalidate, ascending and
// deduplicated: fallback_ubatch, every rung in [fallback_ubatch, cap], and the
// tuning-cache value when it passes llama_auto_ubatch_cached_valid (0 means
// none). Writes at most max_out entries and returns how many. fallback_ubatch
// above cap is left out, as a value the trial cannot try.
inline size_t llama_auto_ubatch_rung_set(const uint32_t * ladder,
                                         size_t           n_ladder,
                                         uint32_t         fallback_ubatch,
                                         uint32_t         cap,
                                         uint32_t         cached_ubatch,
                                         uint32_t *       out,
                                         size_t           max_out) {
    size_t n = 0;
    if (fallback_ubatch <= cap) {
        n = llama_auto_ubatch_set_insert(out, n, max_out, fallback_ubatch);
    }
    for (size_t i = 0; i < n_ladder; ++i) {
        if (ladder[i] >= fallback_ubatch && ladder[i] <= cap) {
            n = llama_auto_ubatch_set_insert(out, n, max_out, ladder[i]);
        }
    }
    if (cached_ubatch != 0 && llama_auto_ubatch_cached_valid(ladder, n_ladder, cached_ubatch, fallback_ubatch, cap)) {
        n = llama_auto_ubatch_set_insert(out, n, max_out, cached_ubatch);
    }
    return n;
}

// The members of `set` that are rungs of `ladder`, in the set's order. The
// ladder loop iterates these: a fallback or cached value that is not a rung
// is in the set for planning only and is never a candidate of its own.
inline size_t llama_auto_ubatch_ladder_members(const uint32_t * set,
                                               size_t           n_set,
                                               const uint32_t * ladder,
                                               size_t           n_ladder,
                                               uint32_t *       out,
                                               size_t           max_out) {
    size_t n = 0;
    for (size_t i = 0; i < n_set && n < max_out; ++i) {
        for (size_t k = 0; k < n_ladder; ++k) {
            if (set[i] == ladder[k]) {
                out[n++] = set[i];
                break;
            }
        }
    }
    return n;
}
