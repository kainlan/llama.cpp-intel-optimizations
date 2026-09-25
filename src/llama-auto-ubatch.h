#pragma once

#include <cstddef>
#include <cstdint>

// Pure helpers behind the SYCL auto micro-batch trial
// (llama_context::sycl_select_auto_ubatch). They take plain integers and
// touch no context or backend state, so tests/test-auto-ubatch-ladder.cpp
// executes them on the host.

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
