#pragma once

// llama.cpp-mmi1: what a context refusal says when a scheduler compute buffer could not be placed on a SYCL device.
// Pure arithmetic and text over plain numbers, so it is host-testable (tests/test-compute-refusal-advice.cpp) and the
// SYCL translation unit only gathers the inputs.

#include <cstddef>
#include <cstdint>
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
    int                    device          = 0;
    uint32_t               n_ubatch        = 0;
    size_t                 request         = 0;  // the refused buffer at n_ubatch
    size_t                 runtime_room    = 0;  // the RUNTIME zone's largest free block, net of the planned scratch hold
    size_t                 kv_room         = 0;  // the KV zone's largest free block
    size_t                 raw_free        = 0;  // the card's free memory outside the arena
    size_t                 headroom_target = 0;  // the driver headroom the arena keeps outside itself
    bool                   hold_fit_refused = false;  // the kpjw hold-spill fit refused n_ubatch ...
    uint32_t               hold_largest_ub  = 0;      // ... and names this -ub (0: none)
    compute_refusal_budget budget;
};

struct compute_refusal_advice {
    size_t   best_room  = 0;  // the largest block any tier could have given the buffer
    uint32_t largest_ub = 0;  // 0: no -ub is known to fit
    int      budget_pct = 0;  // 0: no GGML_SYCL_VRAM_BUDGET_PCT is known to free enough
};

inline size_t compute_refusal_best_room(const compute_refusal_inputs &) {
    return 0;
}

inline size_t compute_refusal_scaled_request(size_t, uint32_t, uint32_t) {
    return 0;
}

inline uint32_t compute_refusal_largest_ub(const compute_refusal_inputs &) {
    return 0;
}

inline size_t compute_refusal_budget_bytes_at(const compute_refusal_budget &, int) {
    return 0;
}

inline int compute_refusal_budget_pct(const compute_refusal_inputs &) {
    return 0;
}

inline compute_refusal_advice compute_refusal_advise(const compute_refusal_inputs &) {
    return {};
}

inline std::string compute_refusal_message(const compute_refusal_inputs &, const compute_refusal_advice &) {
    return std::string();
}

}  // namespace ggml_sycl
