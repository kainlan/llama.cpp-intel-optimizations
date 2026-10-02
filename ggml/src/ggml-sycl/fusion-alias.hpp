#pragma once

// Aliasing gate for fused kernels (llama.cpp-rb2h).
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
// A pair of tensors is admitted when it is disjoint, or when it is the same address with the same type,
// shape and strides AND the kernel reads each element before it writes that same element (an in-place
// pair, `in_place_ok`). Anything else declines the fusion: a partial overlap, the same address with a
// different layout, an identical pair the kernel cannot run in place, or a tensor the caller could not
// resolve. The unfused kernels run one after another and are correct for any layout.
//
// Only what a fused kernel WRITES is an output. An intermediate that the fused kernel never materialises
// (the RMS_NORM result of RMS_NORM+MUL, the MUL result of MUL+ADD) is not written, so it cannot race.
//
// The addresses are the caller's: every operand is resolved once, with the resolver the fused kernel
// itself uses for that operand, so the gate and the kernel cannot disagree about where a tensor lives.

#include "ggml-impl.h"
#include "ggml.h"

#include <cstdint>

enum ggml_sycl_fusion_alias_verdict {
    GGML_SYCL_FUSION_ALIAS_SAFE,
    GGML_SYCL_FUSION_ALIAS_UNRESOLVED,  // a tensor has no resolved address, or the chain is malformed
    GGML_SYCL_FUSION_ALIAS_OVERLAP,     // the byte ranges partially overlap
    GGML_SYCL_FUSION_ALIAS_LAYOUT,      // same address, different type, shape or strides
    GGML_SYCL_FUSION_ALIAS_INPLACE,     // identical layout, but the kernel cannot run in place with it
};

static inline const char * ggml_sycl_fusion_alias_verdict_name(ggml_sycl_fusion_alias_verdict v) {
    switch (v) {
        case GGML_SYCL_FUSION_ALIAS_SAFE:
            return "safe";
        case GGML_SYCL_FUSION_ALIAS_UNRESOLVED:
            return "unresolvable pointer";
        case GGML_SYCL_FUSION_ALIAS_OVERLAP:
            return "partial overlap";
        case GGML_SYCL_FUSION_ALIAS_LAYOUT:
            return "same address, different layout";
        case GGML_SYCL_FUSION_ALIAS_INPLACE:
            return "identical address, kernel not safe in place";
    }
    return "unknown";
}

// The fused sites the gate covers. The chain sites derive their operands from the graph; the two bit0
// sites (MUL_MAT+ADD, router) are not chains and hand their resolved operands to the check directly.
enum ggml_sycl_fusion_alias_site {
    GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD,  // bit1: RMS_NORM, MUL, ADD     -> writes the ADD
    GGML_SYCL_FUSION_SITE_RMS_NORM_MUL,      // bit4: RMS_NORM, MUL          -> writes the MUL
    GGML_SYCL_FUSION_SITE_ADD_RMS_NORM,      // bit5: ADD, RMS_NORM          -> writes both
    GGML_SYCL_FUSION_SITE_MUL_ADD,           // bit6: MUL, ADD               -> writes the ADD
    GGML_SYCL_FUSION_SITE_MUL_MAT_ADD,       // bit0: MUL_MAT + ADD          -> writes the ADD
    GGML_SYCL_FUSION_SITE_ROUTER,            // bit0: router MUL_MAT+ADD+ARGSORT -> writes probs and argsort
    GGML_SYCL_FUSION_SITE_COUNT,
};

static inline const char * ggml_sycl_fusion_alias_site_name(ggml_sycl_fusion_alias_site site) {
    switch (site) {
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD:
            return "RMS_NORM+MUL+ADD";
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL:
            return "RMS_NORM+MUL";
        case GGML_SYCL_FUSION_SITE_ADD_RMS_NORM:
            return "ADD+RMS_NORM";
        case GGML_SYCL_FUSION_SITE_MUL_ADD:
            return "MUL+ADD";
        case GGML_SYCL_FUSION_SITE_MUL_MAT_ADD:
            return "MUL_MAT+ADD";
        case GGML_SYCL_FUSION_SITE_ROUTER:
            return "router";
        case GGML_SYCL_FUSION_SITE_COUNT:
            break;
    }
    return "?";
}

// Chain length of a chain site, 0 for a site that is not a chain.
static inline int ggml_sycl_fusion_alias_site_nodes(ggml_sycl_fusion_alias_site site) {
    switch (site) {
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD:
            return 3;
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL:
        case GGML_SYCL_FUSION_SITE_ADD_RMS_NORM:
        case GGML_SYCL_FUSION_SITE_MUL_ADD:
            return 2;
        default:
            return 0;
    }
}

// Bit k is set when chain node k is written by the fused kernel (see norm.cpp and binbcast.cpp).
static inline unsigned ggml_sycl_fusion_alias_site_written(ggml_sycl_fusion_alias_site site) {
    switch (site) {
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL_ADD:
            return 1u << 2;
        case GGML_SYCL_FUSION_SITE_RMS_NORM_MUL:
            return 1u << 1;
        case GGML_SYCL_FUSION_SITE_ADD_RMS_NORM:
            return (1u << 0) | (1u << 1);
        case GGML_SYCL_FUSION_SITE_MUL_ADD:
            return 1u << 1;
        default:
            return 0;
    }
}

// One resolved operand. `in_place_ok` says the kernel reads this operand's element before it writes the
// same element of an identical output, so an identical pair is admitted. Every write is created with it set:
// two identical outputs are the same element written by the same work-item.
struct ggml_sycl_fusion_operand {
    const ggml_tensor * t;
    const void *        ptr;
    bool                in_place_ok;
};

struct ggml_sycl_fusion_alias_result {
    ggml_sycl_fusion_alias_verdict verdict;
    const ggml_tensor *            write;  // the output that failed, null when safe
    const ggml_tensor *            other;  // what it overlaps, null when safe or malformed
};

static inline ggml_sycl_fusion_alias_verdict ggml_sycl_fusion_alias_classify(const ggml_sycl_fusion_operand & a,
                                                                             const ggml_sycl_fusion_operand & b) {
    if (a.t == nullptr || b.t == nullptr) {
        return GGML_SYCL_FUSION_ALIAS_UNRESOLVED;
    }
    if (ggml_nbytes(a.t) == 0 || ggml_nbytes(b.t) == 0) {
        return GGML_SYCL_FUSION_ALIAS_SAFE;
    }
    if (a.ptr == nullptr || b.ptr == nullptr) {
        return GGML_SYCL_FUSION_ALIAS_UNRESOLVED;
    }
    const uintptr_t a_lo = reinterpret_cast<uintptr_t>(a.ptr);
    const uintptr_t b_lo = reinterpret_cast<uintptr_t>(b.ptr);
    const uintptr_t a_hi = a_lo + ggml_nbytes(a.t);
    const uintptr_t b_hi = b_lo + ggml_nbytes(b.t);
    if (a_hi <= b_lo || b_hi <= a_lo) {
        return GGML_SYCL_FUSION_ALIAS_SAFE;
    }
    if (a_lo != b_lo || a.t->type != b.t->type || !ggml_are_same_shape(a.t, b.t) || !ggml_are_same_stride(a.t, b.t)) {
        return a_lo == b_lo ? GGML_SYCL_FUSION_ALIAS_LAYOUT : GGML_SYCL_FUSION_ALIAS_OVERLAP;
    }
    return (a.in_place_ok && b.in_place_ok) ? GGML_SYCL_FUSION_ALIAS_SAFE : GGML_SYCL_FUSION_ALIAS_INPLACE;
}

// Every write against every later write and every read. The first failing pair is reported.
static inline ggml_sycl_fusion_alias_result ggml_sycl_fusion_alias_check(const ggml_sycl_fusion_operand * writes,
                                                                         int                              n_writes,
                                                                         const ggml_sycl_fusion_operand * reads,
                                                                         int                              n_reads) {
    for (int j = 0; j < n_writes; ++j) {
        for (int k = j + 1; k < n_writes; ++k) {
            const ggml_sycl_fusion_alias_verdict v = ggml_sycl_fusion_alias_classify(writes[j], writes[k]);
            if (v != GGML_SYCL_FUSION_ALIAS_SAFE) {
                return { v, writes[j].t, writes[k].t };
            }
        }
        for (int k = 0; k < n_reads; ++k) {
            const ggml_sycl_fusion_alias_verdict v = ggml_sycl_fusion_alias_classify(writes[j], reads[k]);
            if (v != GGML_SYCL_FUSION_ALIAS_SAFE) {
                return { v, writes[j].t, reads[k].t };
            }
        }
    }
    return { GGML_SYCL_FUSION_ALIAS_SAFE, nullptr, nullptr };
}

// Largest chain a site fuses, and the most operands such a chain can name.
#define GGML_SYCL_FUSION_ALIAS_MAX_CHAIN    4
#define GGML_SYCL_FUSION_ALIAS_MAX_OPERANDS (GGML_SYCL_FUSION_ALIAS_MAX_CHAIN * (GGML_MAX_SRC + 1))

// The check for a chain site. The reads are every source of a chain node that is not itself a chain node;
// the writes are the chain nodes the site's kernel writes. `resolve(t, is_write)` returns the address the
// fused kernel uses for t, and is called once per distinct tensor.
template <typename resolve_fn>
static inline ggml_sycl_fusion_alias_result ggml_sycl_fusion_chain_alias_check(const ggml_cgraph *         cgraph,
                                                                               int                         node_idx,
                                                                               ggml_sycl_fusion_alias_site site,
                                                                               resolve_fn                  resolve) {
    const int      count   = ggml_sycl_fusion_alias_site_nodes(site);
    const unsigned written = ggml_sycl_fusion_alias_site_written(site);
    if (!cgraph || node_idx < 0 || count <= 0 || count > GGML_SYCL_FUSION_ALIAS_MAX_CHAIN ||
        node_idx + count > cgraph->n_nodes) {
        return { GGML_SYCL_FUSION_ALIAS_UNRESOLVED, nullptr, nullptr };
    }
    const auto in_chain = [&](const ggml_tensor * t) {
        for (int k = 0; k < count; ++k) {
            if (cgraph->nodes[node_idx + k] == t) {
                return true;
            }
        }
        return false;
    };

    ggml_sycl_fusion_operand writes[GGML_SYCL_FUSION_ALIAS_MAX_CHAIN];
    ggml_sycl_fusion_operand reads[GGML_SYCL_FUSION_ALIAS_MAX_OPERANDS];
    int                      n_writes = 0;
    int                      n_reads  = 0;

    for (int k = 0; k < count; ++k) {
        if (written & (1u << k)) {
            ggml_tensor * node = cgraph->nodes[node_idx + k];
            writes[n_writes++] = { node, resolve(node, true), true };
        }
    }
    for (int k = 0; k < count; ++k) {
        const ggml_tensor * consumer = cgraph->nodes[node_idx + k];
        for (int s = 0; s < GGML_MAX_SRC && consumer->src[s]; ++s) {
            const ggml_tensor * in = consumer->src[s];
            if (in_chain(in)) {
                continue;
            }
            bool seen = false;
            for (int r = 0; r < n_reads; ++r) {
                seen = seen || reads[r].t == in;
            }
            if (!seen) {
                reads[n_reads++] = { in, resolve(in, false), true };
            }
        }
    }
    return ggml_sycl_fusion_alias_check(writes, n_writes, reads, n_reads);
}
