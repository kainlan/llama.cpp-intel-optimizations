#pragma once

// The tenant element and the cohort ids, in plain C: no std types and no fixed
// underlying type, so a C header (the backend's extern "C" interface) can include
// it. The cohort table that gives each id its name, tier, scope and lifetime is
// in context-tenant-measure.hpp.

#include <stdint.h>

// One slot of the tenant section. Tier, scope and lifetime are not fields:
// the cohort table is their only source, so each fact has one.
// Fields are only appended, and a reader gates each element on struct_size.
struct ggml_sycl_context_tenant_desc {
    uint32_t struct_size;  // element stride gate, as for every section
    uint32_t cohort;       // the cohort id; tier, scope and lifetime are fixed per cohort
    uint32_t slot_index;   // the claim index
    int32_t  device;       // SYCL device index; -1 for the host-pinned tier
    uint64_t slot_bytes;   // the slot's cap
};

// The cohorts this header registers. The value is the id carried in the
// element, so it is append-only; the cohort's name (the `cohort=` text of an
// allocation line) is in the table, and an allocation site reads it from there
// rather than retyping the literal. Each further producer adds its rows here.
enum ggml_sycl_context_cohort {
    GGML_SYCL_CONTEXT_COHORT_COMPUTE = 0,  // device compute chunk slots, index = chunk index
    GGML_SYCL_CONTEXT_COHORT_COMPUTE_HOST =
        1,  // SYCL_Host compute chunk slots, device = -1; one host buft on one device, so demands from several devices merge by maximum
    GGML_SYCL_CONTEXT_COHORT_FATTN_MATERIALIZE = 2,  // the fattn materialization slot, index 0
    GGML_SYCL_CONTEXT_COHORT_NONFA_STAGE       = 3,  // batched-f16 src1 staging slots
    GGML_SYCL_CONTEXT_COHORT_GRAPH_STAGE       = 4,  // staged graph sources, index = k-th staged source
    GGML_SYCL_CONTEXT_COHORT_COUNT
};
