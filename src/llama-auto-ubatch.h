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
// own publish of fallback_ubatch still stands, so only a candidate value left
// in n_ubatch that differs from fallback_ubatch needs one.
inline bool llama_auto_ubatch_settle_needs_publish(bool     published_any,
                                                   bool     publish_dirty,
                                                   uint32_t n_ubatch,
                                                   uint32_t fallback_ubatch) {
    return published_any || publish_dirty || n_ubatch != fallback_ubatch;
}
