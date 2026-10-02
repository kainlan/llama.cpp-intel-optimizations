#pragma once

// Aliasing gate for fused chain kernels (llama.cpp-rb2h).
//
// ggml_gallocr allocates a graph in UNFUSED order: an operand's block is free for reuse as soon as its
// last consumer has run, so a node's output may land on, or straddle, the block of an input that the
// node's own chain still reads. Unfused that is legal, because each op finishes before the next starts.
// A fused kernel reads the chain's external inputs while it writes the chain's outputs, one workgroup
// per row, so an output that overlaps an input at a different offset is a read/write race between
// workgroups: workgroup r writes cols of row r-1 that workgroup r-1 is still reading. The fused
// ADD+RMS_NORM output of a Qwen3.6-27B layer sat 8192 bytes off the ADD operand it overlapped and the
// perplexity varied from run to run.
//
// Two layouts are safe and are admitted: disjoint ranges, and the same address with the same shape and
// strides (an in-place chain, where each thread reads an element before it writes that same element).
// Anything else, and any tensor the caller cannot resolve, declines the fusion; the unfused kernels run
// one after another and are correct for any layout.

#include "ggml-impl.h"
#include "ggml.h"

#include <cstdint>

enum ggml_sycl_fusion_alias {
    GGML_SYCL_FUSION_ALIAS_DISJOINT,
    GGML_SYCL_FUSION_ALIAS_IDENTICAL,
    GGML_SYCL_FUSION_ALIAS_PARTIAL,
};

static inline ggml_sycl_fusion_alias ggml_sycl_fusion_alias_classify(const ggml_tensor * a,
                                                                     const void *        a_ptr,
                                                                     const ggml_tensor * b,
                                                                     const void *        b_ptr) {
    if (ggml_nbytes(a) == 0 || ggml_nbytes(b) == 0) {
        return GGML_SYCL_FUSION_ALIAS_DISJOINT;
    }
    if (a_ptr == nullptr || b_ptr == nullptr) {
        return GGML_SYCL_FUSION_ALIAS_PARTIAL;
    }
    const uintptr_t a_lo = reinterpret_cast<uintptr_t>(a_ptr);
    const uintptr_t b_lo = reinterpret_cast<uintptr_t>(b_ptr);
    const uintptr_t a_hi = a_lo + ggml_nbytes(a);
    const uintptr_t b_hi = b_lo + ggml_nbytes(b);
    if (a_hi <= b_lo || b_hi <= a_lo) {
        return GGML_SYCL_FUSION_ALIAS_DISJOINT;
    }
    if (a_lo == b_lo && a->type == b->type && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b)) {
        return GGML_SYCL_FUSION_ALIAS_IDENTICAL;
    }
    return GGML_SYCL_FUSION_ALIAS_PARTIAL;
}

// `resolve` maps a tensor to the address the fused kernel will use for it.
// Every chain node is an output; every source that is not itself a chain node is an external input.
template <typename resolve_fn>
static inline bool ggml_sycl_fusion_chain_alias_safe(const ggml_cgraph * cgraph,
                                                     int                 node_idx,
                                                     int                 count,
                                                     resolve_fn          resolve) {
    if (!cgraph || node_idx < 0 || count <= 0 || node_idx + count > cgraph->n_nodes) {
        return false;
    }
    const auto in_chain = [&](const ggml_tensor * t) {
        for (int k = 0; k < count; ++k) {
            if (cgraph->nodes[node_idx + k] == t) {
                return true;
            }
        }
        return false;
    };
    for (int j = 0; j < count; ++j) {
        const ggml_tensor * out = cgraph->nodes[node_idx + j];
        const void *        ptr = resolve(out);
        for (int k = j + 1; k < count; ++k) {
            const ggml_tensor * other = cgraph->nodes[node_idx + k];
            if (ggml_sycl_fusion_alias_classify(out, ptr, other, resolve(other)) == GGML_SYCL_FUSION_ALIAS_PARTIAL) {
                return false;
            }
        }
        for (int k = 0; k < count; ++k) {
            const ggml_tensor * consumer = cgraph->nodes[node_idx + k];
            for (int s = 0; s < GGML_MAX_SRC && consumer->src[s]; ++s) {
                const ggml_tensor * in = consumer->src[s];
                if (in_chain(in)) {
                    continue;
                }
                if (ggml_sycl_fusion_alias_classify(out, ptr, in, resolve(in)) == GGML_SYCL_FUSION_ALIAS_PARTIAL) {
                    return false;
                }
            }
        }
    }
    return true;
}
