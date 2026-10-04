// The one list of weight types the unified MUL_MAT kernel serves (llama.cpp-8ony), the one list of weight types the
// layout policy can materialize COALESCED, and the one list of types an MMQ kernel serves (llama.cpp-gldu).
//
// Header-only and free of SYCL, so both the dispatch router (dispatch.hpp, should_use_unified) and the zone planner's
// adapter (unified-cache.cpp) can include it. The planner needs the answer before any graph exists: it reserves the
// planned dequant buffers for a weight the unified kernel's oneDNN f16 route can draw, so a type added here without
// the planner seeing it would be drawn at run time and never planned. Keeping one list removes that drift.
#pragma once

#include "ggml.h"

namespace ggml_sycl {

// Unified kernel supports the block-quantized types Q4_0 and MXFP4. FP16/BF16 use oneDNN.
inline bool unified_kernel_serves_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_MXFP4:
            // TODO: Add Q8_0, Q6_K, Q4_K support to unified kernel
            return true;
        default:
            return false;  // FP16, BF16, F32, Q6_K, etc. use legacy path for now
    }
}

// The weight types the layout policy can materialize in the COALESCED layout. Add a type here only when its coalesced
// kernels exist. Read by is_coalesced_supported (common.hpp) and by the zone planner, through
// dense_pp_route_is_f16_dequant_arm.
inline bool coalesced_capable_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

// The weight types an MMQ kernel serves. Read by ggml_sycl_supports_mmq (the router's eligibility term) and, through
// dense_pp_route_is_f16_dequant_arm, by the zone planner.
inline bool mmq_capable_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// The types SYCL executes as a dense MUL_MAT operand: the allowlist behind supports_op for MUL_MAT (and the dense
// ADD_ID operand). Read by ggml_sycl_mul_mat_type_supported (ggml-sycl.cpp) and by the zone planner, which must not
// reserve a dequant copy for a type SYCL refuses (NVFP4, Q1_0, Q2_0: they run on ggml-cpu).
inline bool dense_mul_mat_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// Whether a dense QUANTIZED MUL_MAT weight of this type takes the oneDNN f16 dequant arm as its PP route
// (llama.cpp-gldu). The caller has already established that the type is quantized and the operand a dense weight.
//
// The router walks k_mul_mat_priority (ggml-sycl.cpp): a type an MMQ kernel serves is taken by it at PP batch (MMQ_AOS
// precedes ONEDNN_AOS, and use_mmq holds up to MMQ_MAX_BATCH_SIZE), a type with a coalesced layout is taken by its
// coalesced kernel, the oneDNN coalesced/SOA arm (Q8_0, which has its own unconditional plan mark) or the unified kernel
// (Q4_0 / MXFP4, which have the unified-kernel plan mark). A quantized type left after those lists, the IQ family, has
// no kernel but the dequant arm at PP batch, so for it the arm is the planned route and not a fallback.
//
// The planner cannot ask the router: it has no graph node, no batch and no resolved layout, so it asks the two type
// lists the router's own eligibility terms are built from. The answer is a type-level superset the router can only
// narrow (an op the oneDNN PP scratch supplies, a batch within the MMVQ limit). A batch above MMQ_MAX_BATCH_SIZE sends
// an MMQ type to the arm too; that residue is left to the graph-entry walk, which grows the buffer or refuses by name.
inline bool dense_pp_route_is_f16_dequant_arm(ggml_type type) {
    return !coalesced_capable_type(type) && !mmq_capable_type(type);
}

// Whether the planner reserves the dense f16 dequant buffers for this weight because the dequant arm is its PP route
// (llama.cpp-gldu), for the types the unified kernel's own mark does not cover. ONE function so the adapter's
// composition is testable without a device:
//   - quantized, and a type no MMQ, coalesced or unified kernel serves (dense_pp_route_is_f16_dequant_arm);
//   - a type SYCL executes as a dense MUL_MAT at all (dense_mul_mat_type_supported);
//   - one the router can dequantize (dequant_supported, from onednn_woq::supports_dequant_fp16, which the SYCL-free
//     header cannot call);
//   - not an expert stack (MUL_MAT_ID keeps its own workspace);
//   - not the LM head (is_head, from the usage classification and the tied-embedding classifier): a completion runs
//     it on the last row of each ubatch, so it normally never reaches the arm, and claiming its f16 copy (1.2 GB for
//     a 248k-vocabulary head) would size the whole plan from it. Perplexity and embeddings run it on many rows and
//     are covered by the graph-entry walk, which grows the buffer or refuses by name. This is deliberately unlike the
//     Q4_0 / MXFP4 head, which stays planned unconditionally by owner decision (llama.cpp-8ony) until llama.cpp-fkpg
//     delivers n_outputs to the planner; that gate will reinstate both.
inline bool aos_dequant_f16_plan_claims(ggml_type type,
                                        bool      quantized,
                                        bool      is_head,
                                        bool      is_expert,
                                        bool      dequant_supported) {
    return quantized && !is_head && !is_expert && dequant_supported && dense_mul_mat_type_supported(type) &&
           dense_pp_route_is_f16_dequant_arm(type);
}

}  // namespace ggml_sycl
