#pragma once

// Aliasing gate for fused chain kernels (llama.cpp-rb2h).
//
// RED STUB: admits every chain, which is what the fusion sites did before the gate existed.

#include "ggml-impl.h"
#include "ggml.h"

template <typename resolve_fn>
static inline bool ggml_sycl_fusion_chain_alias_safe(const ggml_cgraph * cgraph,
                                                     int                 node_idx,
                                                     int                 count,
                                                     resolve_fn          resolve) {
    (void) cgraph;
    (void) node_idx;
    (void) count;
    (void) resolve;
    return true;
}
