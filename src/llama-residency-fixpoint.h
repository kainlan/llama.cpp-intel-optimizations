#pragma once

// The pure parts of the constructor's planned reserve (llama.cpp-7gno, zhcn design 2.4 and 2.7):
//
//   llama_plan_caps_decide()     whether a context acquires its chunk-cap copy (`plan_caps`), runs
//                                unplanned, or is refused by name;
//   llama_residency_fixpoint()   the demote-only residency iteration, written over the probe and the
//                                measure it is given, so the constructor passes the backend's and a
//                                host test passes stubs.
//
// Both are production-unreachable: the constructor reaches the decision's ACQUIRE arm only when the
// residency probe is wired (llama.cpp-7gno), and the probe is moua's tenant-aware L4 probe, which does
// not exist yet. The fixpoint is the pure iteration; the constructor passes the backend's probe and
// measure, a host test passes stubs.

#include "ggml-sycl-cohort.h"
#include "ggml-sycl.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
// refusal becomes reachable when moua L4 step 3 lands; no code change needed here. (The caller's
// `l4_available` is also false until the residency probe is wired, so the landing of the three procs
// alone changes nothing: see llama_context_l4_ready in llama-context.cpp.)
inline llama_plan_caps_decision llama_plan_caps_decide(bool has_sycl_backend,
                                                       bool plan_active,
                                                       bool new_present,
                                                       bool free_present,
                                                       bool l4_available) {
    if (!has_sycl_backend || !plan_active || !l4_available) {
        return LLAMA_PLAN_CAPS_UNPLANNED;
    }
    return new_present && free_present ? LLAMA_PLAN_CAPS_ACQUIRE : LLAMA_PLAN_CAPS_REFUSE_NO_PROCS;
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
};

struct llama_residency_fixpoint_result {
    llama_residency_fixpoint_status status = LLAMA_RESIDENCY_FIXPOINT_OK;
    llama_residency                 residency;              // R*
    llama_tenants                   tenants;                // T*
    uint32_t                        iterations    = 0;      // MEASUREs of the iteration, the verify's excluded
    bool                            verify_shrunk = false;  // the verify replaced (R*, T*) with (R_v, T_v)
    std::string                     reason;
};

// The residency fixpoint (design 2.7), demote-only:
//
//   R0 := probe(none)
//   repeat: T := measure(R); R := R union probe(T)   until R stops growing   (at most n_layer + 1 rounds)
//   verify: R_v := probe(T*) without the union; when R_v is strictly smaller than R*, measure over R_v
//           and take (R_v, T_v) iff the probe over T_v keeps R_v
//
//   probe(tenants)         the residency the backend would give with those tenants (nullptr: none)
//   measure(R, tenants)    the tenant section measured over memory created with residency R;
//                          false when the measure cannot be made
//
// The union makes R grow by at least one layer per round or stop, so n_layer + 1 rounds always
// suffice. A verify answer that holds a host layer R* does not contradicts that, and is a BUG.
template <typename Probe, typename Measure>
llama_residency_fixpoint_result llama_residency_fixpoint(size_t n_layer, Probe probe, Measure measure) {
    llama_residency_fixpoint_result out;

    const auto bug = [&](const char * why) {
        out.status = LLAMA_RESIDENCY_FIXPOINT_BUG;
        out.reason = why;
        return out;
    };
    const auto measure_failed = [&]() {
        out.status = LLAMA_RESIDENCY_FIXPOINT_MEASURE_FAILED;
        out.reason = "the tenant section could not be measured";
        return out;
    };

    llama_residency residency = probe(nullptr);
    if (residency.size() != n_layer) {
        return bug("the residency probe answered a residency of the wrong length");
    }

    llama_tenants tenants;
    for (size_t round = 0; round <= n_layer + 1; round++) {
        if (!measure(residency, tenants)) {
            return measure_failed();
        }
        out.iterations++;

        const llama_residency answer = probe(&tenants);
        if (answer.size() != n_layer) {
            return bug("the residency probe answered a residency of the wrong length");
        }
        llama_residency next = residency;
        for (size_t i = 0; i < n_layer; i++) {
            next[i] = next[i] || answer[i];
        }
        if (next == residency) {
            break;
        }
        residency = next;
    }

    // the verify: the probe over T* without the union
    const llama_residency raw = probe(&tenants);
    if (raw.size() != n_layer) {
        return bug("the residency probe answered a residency of the wrong length");
    }
    bool shrinks = false;
    for (size_t i = 0; i < n_layer; i++) {
        if (raw[i] && !residency[i]) {
            return bug("the verify holds a host layer the fixpoint does not");
        }
        shrinks = shrinks || (!raw[i] && residency[i]);
    }
    if (shrinks) {
        llama_tenants shrunk;
        if (!measure(raw, shrunk)) {
            return measure_failed();
        }
        if (probe(&shrunk) == raw) {
            residency         = raw;
            tenants           = shrunk;
            out.verify_shrunk = true;
        }
    }

    out.residency = residency;
    out.tenants   = tenants;
    return out;
}
