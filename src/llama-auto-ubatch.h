#pragma once

#include <cstddef>
#include <cstdint>

// Pure helpers behind the SYCL auto micro-batch trial
// (llama_context::sycl_select_auto_ubatch). They take plain integers and
// touch no context or backend state, so tests/test-auto-ubatch-ladder.cpp
// executes them on the host.

// The trial's candidate micro-batch sizes, ascending.
static const uint32_t llama_auto_ubatch_ladder[] = { 512, 1024, 2048, 4096 };
static const size_t   llama_auto_ubatch_ladder_size =
    sizeof(llama_auto_ubatch_ladder) / sizeof(llama_auto_ubatch_ladder[0]);

// The smallest n_ubatch the trial lowers itself to (llama.cpp-kpjw).
static const uint32_t llama_auto_ubatch_descent_floor = 64;

// The next rung of the trial's downward continuation: half of `from`, rounded down to a multiple of 32, or 0 when
// that is under llama_auto_ubatch_descent_floor (nothing below it) or `from` is 0. When the default rung, and so
// every rung above it, is refused, the trial lowers n_ubatch rather than refusing the context: a smaller -ub is not
// a smaller context (B50, Qwen3.6-27B auto: 512 spilled a 495 MB compute buffer outside the arena and left 107.7 MB
// against the 256 MB driver headroom; 256 spills half of that and fits).
inline uint32_t llama_auto_ubatch_next_lower(uint32_t from) {
    const uint32_t half = from / 2;
    return half >= llama_auto_ubatch_descent_floor ? half - half % 32 : 0;
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
