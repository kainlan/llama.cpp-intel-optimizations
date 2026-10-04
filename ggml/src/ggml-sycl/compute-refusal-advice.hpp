#pragma once

// llama.cpp-mmi1: what a context refusal says when a scheduler compute buffer could not be placed on a SYCL device.
// Pure arithmetic and text over plain numbers, so it is host-testable (tests/test-compute-refusal-advice.cpp) and the
// SYCL translation unit only gathers the inputs.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

namespace ggml_sycl {

// The budget authority's published figures for the device (compute_vram_budget_authority()).
struct compute_refusal_budget {
    int    pct               = 100;  // resolved GGML_SYCL_VRAM_BUDGET_PCT
    size_t base_mem          = 0;    // host-unified-adjusted device total; 0 when unknown
    size_t budget_bytes      = 0;    // the arena budget now: min(base*pct/100, free at init) - external headroom
    size_t external_headroom = 0;    // the headroom the authority subtracted
    size_t fixed_zone_bytes  = 0;    // the arena zones that are not weights (RUNTIME, SCRATCH, ONEDNN)
};

struct compute_refusal_inputs {
    int                    device       = 0;
    uint32_t               n_ubatch     = 0;
    size_t                 request      = 0;  // the refused buffer at n_ubatch
    size_t                 runtime_room = 0;  // the RUNTIME zone's largest free block
    size_t                 kv_room      = 0;  // the KV zone's largest free block
    size_t                 raw_free     = 0;  // the card's free memory outside the arena
    size_t                 headroom_target  = 0;      // the driver headroom the arena keeps outside itself
    bool                   hold_fit_refused = false;  // the kpjw hold-spill fit refused n_ubatch ...
    uint32_t               hold_largest_ub  = 0;      // ... and names this -ub (0: none)
    compute_refusal_budget budget;
};

struct compute_refusal_advice {
    size_t   best_room  = 0;  // the largest block any tier could have given the buffer
    uint32_t largest_ub = 0;  // 0: no -ub is known to fit
    int      budget_pct = 0;  // 0: no GGML_SYCL_VRAM_BUDGET_PCT is known to free enough
};

// The largest block any tier could have given the buffer. A buffer is one block, so the tiers are alternatives, never a
// sum; the raw tier is the card's free memory outside the arena less the driver headroom the arena keeps there.
inline size_t compute_refusal_best_room(const compute_refusal_inputs & in) {
    const size_t raw_room = in.raw_free > in.headroom_target ? in.raw_free - in.headroom_target : 0;
    return std::max(in.runtime_room, std::max(in.kv_room, raw_room));
}

// A compute buffer grows about linearly with n_ubatch (the same assumption zone_hold_fit_request makes), so the buffer
// refused at `from_ub` is scaled to `to_ub`, rounded up and saturating. An unknown source -ub scales nothing.
inline size_t compute_refusal_scaled_request(size_t request, uint32_t from_ub, uint32_t to_ub) {
    if (from_ub == 0 || to_ub == from_ub || to_ub == 0) {
        return request;
    }
    const size_t max = std::numeric_limits<size_t>::max();
    const size_t q   = request / from_ub;
    const size_t r   = request % from_ub;
    if (q > max / to_ub) {
        return max;
    }
    const size_t hi = q * to_ub;
    const size_t lo = (r * to_ub + from_ub - 1) / from_ub;  // r < from_ub, so r * to_ub fits in 64 bits
    return hi > max - lo ? max : hi + lo;
}

// The largest power of two, at least 32 and strictly under the refused n_ubatch, whose scaled buffer fits the best room
// and that the kpjw hold-spill fit also accepts (its own largest -ub caps the answer when it refused the rung, so the
// number printed passes both checks). Never the refused -ub itself. 0: none is known to fit.
inline uint32_t compute_refusal_largest_ub(const compute_refusal_inputs & in) {
    if (in.n_ubatch < 2) {
        return 0;
    }
    const uint32_t cap  = in.hold_fit_refused ? in.hold_largest_ub : std::numeric_limits<uint32_t>::max();
    const size_t   room = compute_refusal_best_room(in);
    uint32_t       p    = 1;
    while (p < in.n_ubatch / 2 + in.n_ubatch % 2 && p * 2 < in.n_ubatch) {
        p *= 2;
    }
    for (; p >= 32; p /= 2) {
        if (p <= cap && compute_refusal_scaled_request(in.request, in.n_ubatch, p) <= room) {
            return p;
        }
    }
    return 0;
}

// The arena budget at `pct`, by the budget authority's own arithmetic (compute_vram_budget_authority()):
// min(base * pct / 100, free at init) - external headroom. The free memory the authority started from is not
// published; budget_bytes + external_headroom is it when that cap bound, and is at least base * pct / 100 otherwise,
// which is the only case in which the cap does not matter, so the reconstruction is exact either way.
inline size_t compute_refusal_budget_bytes_at(const compute_refusal_budget & b, int pct) {
    if (b.base_mem == 0) {
        return 0;
    }
    pct                       = std::max(1, std::min(100, pct));
    size_t       budget       = static_cast<size_t>(b.base_mem * (static_cast<double>(pct) / 100.0));
    const size_t free_at_init = b.budget_bytes + b.external_headroom;
    if (free_at_init > 0 && budget > free_at_init) {
        budget = free_at_init;
    }
    return budget > b.external_headroom ? budget - b.external_headroom : 0;
}

// The highest GGML_SYCL_VRAM_BUDGET_PCT, under the current one, that leaves the refused buffer, the driver headroom and
// one more headroom of margin (another consumer outside the arena) in the card's free memory outside it: every byte
// the budget gives up is a byte outside the arena. 0 when lowering the budget cannot help: the room was already
// there (another cause), the shortfall is more than the whole budget, the pct would starve the arena's fixed zones, or
// the budget authority is unknown.
inline int compute_refusal_budget_pct(const compute_refusal_inputs & in) {
    const compute_refusal_budget & b = in.budget;
    if (b.base_mem == 0 || b.budget_bytes == 0 || in.request == 0) {
        return 0;
    }
    const size_t need = in.request + 2 * in.headroom_target;
    if (in.raw_free >= need) {
        return 0;
    }
    const size_t shortfall = need - in.raw_free;
    if (shortfall >= b.budget_bytes) {
        return 0;
    }
    const size_t target = b.budget_bytes - shortfall;
    for (int p = std::min(b.pct, 100) - 1; p >= 1; --p) {
        const size_t at = compute_refusal_budget_bytes_at(b, p);
        if (at <= target) {
            return at > b.fixed_zone_bytes ? p : 0;
        }
    }
    return 0;
}

inline compute_refusal_advice compute_refusal_advise(const compute_refusal_inputs & in) {
    compute_refusal_advice a;
    a.best_room  = compute_refusal_best_room(in);
    a.largest_ub = compute_refusal_largest_ub(in);
    a.budget_pct = compute_refusal_budget_pct(in);
    return a;
}

// The refusal's text: the request, the room each tier had, why host memory is no answer, and what fits. Never a smaller
// context: the owner's rulings place KV, they do not shrink it.
inline std::string compute_refusal_message(const compute_refusal_inputs & in, const compute_refusal_advice & adv) {
    const double mib = 1024.0 * 1024.0;
    char         buf[1536];
    snprintf(buf, sizeof(buf),
             "the scheduler's compute buffer of %.1f MiB for SYCL device %d at -ub %u fits no tier of the card: the "
             "RUNTIME zone has %.1f MiB free for it, the KV zone %.1f MiB, and the card %.1f MiB outside the arena "
             "once its %.1f MiB driver headroom is kept (a buffer is one block, so the tiers are not added). It is "
             "not placed in host memory instead: a host-pinned buffer is no home for a device compute buffer, the "
             "executor follows placement",
             in.request / mib, in.device, in.n_ubatch, in.runtime_room / mib, in.kv_room / mib,
             (in.raw_free > in.headroom_target ? in.raw_free - in.headroom_target : 0) / mib, in.headroom_target / mib);
    std::string msg = buf;
    if (adv.largest_ub != 0) {
        snprintf(buf, sizeof(buf),
                 "; the largest -ub that fits is -ub %u (a power of two, estimated by scaling this buffer with -ub)",
                 adv.largest_ub);
    } else {
        snprintf(buf, sizeof(buf), "; no -ub is known to fit");
    }
    msg += buf;
    if (adv.budget_pct != 0) {
        snprintf(buf, sizeof(buf),
                 "; or load the model with GGML_SYCL_VRAM_BUDGET_PCT=%d (now %d), which leaves enough of the card "
                 "outside the arena for this buffer at -ub %u",
                 adv.budget_pct, in.budget.pct, in.n_ubatch);
    } else {
        snprintf(buf, sizeof(buf), "; no GGML_SYCL_VRAM_BUDGET_PCT is known to free enough: free VRAM on the card");
    }
    msg += buf;
    return msg;
}

}  // namespace ggml_sycl
