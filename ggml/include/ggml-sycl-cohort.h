#pragma once

// The cohort ids of the measured-tenant section, in plain C and in the public include
// directory: the backend's cohort table (ggml/src/ggml-sycl/context-tenant-measure.cpp) and
// llama's section builder (src/llama-context-tenant.h) both name them, and neither can see
// the other's private headers. The tenant element itself is ggml_sycl_context_tenant_desc,
// defined once in ggml-sycl.h.
//
// The value is the id carried in the element, so it is append-only. The cohort's name (the
// `cohort=` text of an allocation line), tier, scope and lifetime are in the backend's table,
// which is their only source; an id is never retyped as a literal.

enum ggml_sycl_context_cohort {
    GGML_SYCL_CONTEXT_COHORT_COMPUTE           = 0,  // device compute chunk slots, index = chunk index
    GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST      = 1,  // SYCL_Host compute chunk slots, device = -1
    GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE = 2,  // the fattn materialization slot, index 0
    GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE       = 3,  // batched-f16 src1 staging slots
    GGML_SYCL_CONTEXT_COHORT_GRAPH_STAGE       = 4,  // staged graph sources, index = k-th staged source
    GGML_SYCL_CONTEXT_COHORT_COUNT
};
