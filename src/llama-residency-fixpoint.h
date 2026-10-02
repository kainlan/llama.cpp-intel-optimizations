#pragma once

// The pure parts of the constructor's planned reserve (llama.cpp-7gno, zhcn design 2.4 and 2.7):
//
//   llama_plan_caps_decide()     whether a context acquires its chunk-cap copy (`plan_caps`), runs
//                                unplanned, or is refused by name;
//   llama_residency_fixpoint()   the demote-only residency iteration, written over the probe and the
//                                measure it is given, so the constructor passes the backend's and a
//                                host test passes stubs.
//
// Both are production-unreachable until 7gno wires them into the constructor, and the fixpoint's
// probe is moua's tenant-aware L4 probe, which does not exist yet.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-sycl.h"
#include "ggml-sycl-cohort.h"

enum llama_plan_caps_decision {
    LLAMA_PLAN_CAPS_UNPLANNED,
    LLAMA_PLAN_CAPS_ACQUIRE,
    LLAMA_PLAN_CAPS_REFUSE_NO_PROCS,
};

// The text of the refusal. Greppable on purpose: the constructor throws exactly this.
inline const char * llama_plan_caps_missing_procs_reason() {
    return "SYCL placement plan active but ggml_backend_sycl_plan_caps_new/_free is not exported";
}

// Whether a context acquires its chunk-cap copy, runs unplanned, or is refused by name.
//
//   no SYCL backend, no active placement plan, or L4 not all present   -> UNPLANNED
//   otherwise, both cap procs present                                  -> ACQUIRE
//   otherwise                                                          -> REFUSE_NO_PROCS
//
// The L4 test comes first on purpose: while any of the three L4 procs is null the cap procs are never
// consulted, so a backend that predates L4 keeps its legacy publish path whatever it exports. The
// refusal becomes reachable when moua L4 step 3 lands; no code change needed here.
//
// RED stub (llama.cpp-7gno): the real rule lands with the constructor wiring.
inline llama_plan_caps_decision llama_plan_caps_decide(bool has_sycl_backend,
                                                       bool plan_active,
                                                       bool new_present,
                                                       bool free_present,
                                                       bool l4_available) {
    (void) has_sycl_backend;
    (void) plan_active;
    (void) new_present;
    (void) free_present;
    (void) l4_available;
    return LLAMA_PLAN_CAPS_UNPLANNED;
}

// One byte per layer, 1 = the layer is host-resident. Demotion only ever sets a byte.
using llama_residency = std::vector<uint8_t>;
using llama_tenants   = std::vector<ggml_sycl_context_tenant_desc>;

enum llama_residency_fixpoint_status {
    LLAMA_RESIDENCY_FIXPOINT_OK,
    LLAMA_RESIDENCY_FIXPOINT_MEASURE_FAILED,
    // the probe contradicted the iteration: a residency of the wrong length, or a verify answer
    // that holds a host layer the fixpoint did not. A plan bug, never a fit verdict.
    LLAMA_RESIDENCY_FIXPOINT_BUG,
    LLAMA_RESIDENCY_FIXPOINT_NOT_IMPLEMENTED,
};

struct llama_residency_fixpoint_result {
    llama_residency_fixpoint_status status       = LLAMA_RESIDENCY_FIXPOINT_NOT_IMPLEMENTED;
    llama_residency                 residency;       // R*
    llama_tenants                   tenants;         // T*
    uint32_t                        iterations   = 0;      // MEASUREs of the iteration, the verify's excluded
    bool                            verify_shrunk = false;  // the verify replaced (R*, T*) with (R_v, T_v)
    std::string                     reason;
};

// RED stub (llama.cpp-7gno).
//
//   probe(tenants)         the residency the backend would give with those tenants (nullptr: none)
//   measure(R, tenants)    the tenant section measured over memory created with residency R;
//                          false when the measure cannot be made
template <typename Probe, typename Measure>
llama_residency_fixpoint_result llama_residency_fixpoint(size_t n_layer, Probe probe, Measure measure) {
    (void) n_layer;
    (void) probe;
    (void) measure;
    return {};
}
