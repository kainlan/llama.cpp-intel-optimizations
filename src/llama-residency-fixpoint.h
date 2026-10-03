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
// residency probe is wired (llama.cpp-hdpd), and the probe is moua's tenant-aware L4 probe, which does
// not exist yet.

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
// consulted, so a backend that predates L4 keeps its legacy publish path whatever it exports. That
// is a deliberate narrowing of design 2.4's "under an active plan a missing _new/_free is the named
// refusal": the refusal is reachable only once L4 is. The L4 test comes off when the latch does.
//
// The refusal does NOT become reachable when moua's L4 step 3 lands by itself: the caller's
// `l4_available` is also false until the residency probe is wired (llama_context_l4_ready in
// llama-context.cpp, behind the interim latch llama_context_residency_probe_wired). Reaching it takes
// one more edit, llama.cpp-hdpd's: name the probe proc, add it to llama_sycl_l4_procs::available() and
// delete the latch together.
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

// What the residency probe says about itself. The value-initialised answer is NOT_ANSWERED, so a probe that
// returns `{}`, or a wrapper that forgets to set a status, refuses: the fixpoint treats every status other than
// OK -- an unknown value included -- as "the backend could not answer".
//
// Mapping obligation of the adapter (llama.cpp-hdpd / step 1d) that turns the backend's
// ggml_backend_sycl_probe_residency status into this one: OK maps to OK, and every other value, GEOMETRY_NOT_WIRED
// and NOT_ANSWERED both included and any value the adapter does not know, maps to a status other than OK, with the
// backend's reason. It must NEVER answer an all-zero residency in their place: "zero host layers" reads as "every
// layer is device-resident", the answer that fits an unanswered probe least (llama.cpp-71hq).
enum llama_residency_probe_status {
    LLAMA_RESIDENCY_PROBE_NOT_ANSWERED = 0,  // the probe produced nothing (also the default)
    LLAMA_RESIDENCY_PROBE_OK,                // `residency` is the backend's answer
    LLAMA_RESIDENCY_PROBE_REFUSED,           // the backend answered that it cannot (geometry not wired, ...)
};

struct llama_residency_probe_answer {
    llama_residency_probe_status status = LLAMA_RESIDENCY_PROBE_NOT_ANSWERED;
    llama_residency              residency;  // read only when status is OK
    std::string                  reason;     // the backend's reason when it is not
};

enum llama_residency_fixpoint_status {
    LLAMA_RESIDENCY_FIXPOINT_OK,
    LLAMA_RESIDENCY_FIXPOINT_MEASURE_FAILED,
    // the probe contradicted the iteration: a residency of the wrong length, or a loop that did not
    // stop. A plan bug, never a fit verdict.
    LLAMA_RESIDENCY_FIXPOINT_BUG,
    // the probe could not answer (a status other than OK) at the initial call, a round or the verify. Not a bug and
    // not a fit verdict either: the plan was never checked, so it is refused.
    LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED,
};

struct llama_residency_fixpoint_result {
    llama_residency_fixpoint_status status = LLAMA_RESIDENCY_FIXPOINT_OK;
    llama_residency                 residency;              // R*
    llama_tenants                   tenants;                // T*
    uint32_t                        iterations    = 0;      // MEASUREs of the iteration, the verify's excluded
    bool                            verify_shrunk = false;  // the verify replaced (R*, T*) with (R_v, T_v)
    std::string                     reason;
};

// Whether a result is a refusal. Fail-closed: only OK is not, so a status this build does not know is refused too.
inline bool llama_residency_fixpoint_refused(llama_residency_fixpoint_status status) {
    return status != LLAMA_RESIDENCY_FIXPOINT_OK;
}

// The text of the refusal, greppable on purpose: the constructor (llama.cpp-hdpd's consumer of the fixpoint) throws
// exactly this for any result that is refused, and never acquires a plan from one. Empty for an OK result.
inline std::string llama_residency_fixpoint_refusal_text(const llama_residency_fixpoint_result & result) {
    if (!llama_residency_fixpoint_refused(result.status)) {
        return std::string();
    }
    return "SYCL residency fixpoint refused: " + result.reason;
}

// The residency fixpoint (design 2.7), demote-only:
//
//   R0 := probe(none)
//   repeat: T := measure(R); R := R union probe(T)   until R stops growing   (at most n_layer + 1 rounds)
//   verify: R_v := probe(T*) without the union; when R_v is strictly smaller than R*, measure over R_v
//           and take (R_v, T_v) iff the probe over T_v keeps R_v
//
// Design 2.7 writes the verify as `R_v := probe(tenants=T*).residency without the union`. That is the
// answer the last round already got (the iteration stops on the round whose tenants are T*), so it is
// reused, not asked again: a second probe over the same tenants costs one more L0 transaction per
// planned context construction and can differ only if the probe is not deterministic, the contract the
// whole iteration rests on. A byte of a probe's answer other than 0 is host-resident and is normalised
// to 1, so that the comparisons below are about layers.
//
//   probe(tenants)         a llama_residency_probe_answer: the residency the backend would give with those
//                          tenants (nullptr: none), or the status that says it could not give one. A status other
//                          than OK ends the fixpoint at that call as PROBE_FAILED, whatever the answer holds
//   measure(R, tenants)    the tenant section measured over memory created with residency R;
//                          false when the measure cannot be made
//
// The union makes R grow by at least one layer per round or stop, so n_layer + 1 rounds always
// suffice: R0 may be empty, n_layer rounds then grow it to every layer, and one more sees it stop. A loop
// that ends without stopping contradicts that and is a BUG, never a fit verdict. (It cannot happen with
// a probe that answers a residency of n_layer bytes, which is checked, so that arm guards the loop's own
// invariant.)
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

    const auto normalise = [](llama_residency r) {
        for (auto & b : r) {
            b = b != 0 ? 1 : 0;
        }
        return r;
    };

    // The one place the probe is called. A status other than OK refuses at that call: the residency such an answer
    // holds is never read, so a probe that cannot answer cannot read as "every layer is device-resident". A wrong
    // length is a different fault (the probe answered and contradicted the iteration) and stays BUG.
    const auto ask = [&](const llama_tenants * with, const char * site, llama_residency & into) {
        llama_residency_probe_answer a = probe(with);
        if (a.status != LLAMA_RESIDENCY_PROBE_OK) {
            out.status = LLAMA_RESIDENCY_FIXPOINT_PROBE_FAILED;
            out.reason = std::string("the residency probe did not answer (") + site +
                         "): " + (a.reason.empty() ? "no reason given" : a.reason);
            return false;
        }
        if (a.residency.size() != n_layer) {
            bug("the residency probe answered a residency of the wrong length");
            return false;
        }
        into = normalise(std::move(a.residency));
        return true;
    };

    llama_residency residency;
    if (!ask(nullptr, "initial call", residency)) {
        return out;
    }

    llama_tenants   tenants;
    llama_residency answer;  // the last round's answer: over T*, the verify's raw probe
    bool            converged = false;
    for (size_t round = 0; round < n_layer + 1; round++) {
        if (!measure(residency, tenants)) {
            return measure_failed();
        }
        out.iterations++;

        if (!ask(&tenants, "round", answer)) {
            return out;
        }
        llama_residency next = residency;
        for (size_t i = 0; i < n_layer; i++) {
            next[i] = next[i] || answer[i];
        }
        if (next == residency) {
            converged = true;
            break;
        }
        residency = next;
    }
    if (!converged) {
        return bug("the residency iteration did not stop within n_layer + 1 rounds");
    }

    // the verify: the probe over T* without the union, which the last round answered (the loop stopped on
    // it, so it holds no layer R* lacks)
    const llama_residency & raw     = answer;
    bool                    shrinks = false;
    for (size_t i = 0; i < n_layer; i++) {
        shrinks = shrinks || (!raw[i] && residency[i]);
    }
    if (shrinks) {
        llama_tenants shrunk;
        if (!measure(raw, shrunk)) {
            return measure_failed();
        }
        llama_residency kept;
        if (!ask(&shrunk, "verify", kept)) {
            return out;
        }
        if (kept == raw) {
            residency         = raw;
            tenants           = shrunk;
            out.verify_shrunk = true;
        }
    }

    out.residency = residency;
    out.tenants   = tenants;
    return out;
}
