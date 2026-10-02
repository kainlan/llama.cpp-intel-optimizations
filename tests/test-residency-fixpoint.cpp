// llama.cpp-7gno: the constructor's planned-reserve decisions, pure host (zhcn design 2.4, 2.7).
// It links only src/llama-residency-fixpoint.h and the public backend header: no model, no device,
// no scheduler.
//
// Cases:
//   (1) plan_caps decision: a context with no SYCL backend, or with no active placement plan, runs
//       unplanned whatever else is present;
//   (2) inert until L4: an active plan with the chunk-cap procs present but the L4 publish, coverage
//       and late-check procs not all present is UNPLANNED, and that holds with the cap procs missing
//       too (the refusal below is reachable only once L4 is);
//   (3) an active plan with L4 present and both cap procs acquires; with either cap proc missing it
//       is the named refusal, and the refusal text is the literal the design pins;
//   (4) the fixpoint with a probe that never demotes: one MEASURE, R* == R0, T* the measured section;
//   (5) a demotion: the probe under T0 adds a host layer, the second MEASURE runs over that residency,
//       and the loop stops when the probe adds nothing;
//   (6) demote-only: a probe that would promote a layer back (answers a smaller host set) does not
//       shrink R during the iteration;
//   (7) the worst chain, one new host layer per iteration, converges in n_layer + 1 MEASUREs;
//   (8) a MEASURE that cannot be made is MEASURE_FAILED, carries the reason, and publishes nothing;
//   (9) a probe that answers a residency of the wrong length is BUG, never a guess;
//   (10) the verify: a raw probe over T* that holds fewer host layers than R* re-measures over that
//        smaller set and the pair replaces (R*, T*) only when the probe over the new tenants keeps it;
//        when the probe does not keep it, (R*, T*) stands;
//   (11) a verify answer that holds a host layer R* does not is BUG: R* was not a fixpoint.

#include "../src/llama-residency-fixpoint.h"
#include "ggml-sycl-cohort.h"  // GGML_SYCL_CONTEXT_COHORT_COMPUTE, named here, not reached transitively

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_failed++;                                                          \
        }                                                                        \
    } while (0)

// a failed precondition is counted and ends the case, so a missing result reads as FAIL, not a crash
#define REQUIRE(cond)                                                            \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_failed++;                                                          \
            return;                                                              \
        }                                                                        \
    } while (0)

// a tenant section that names the residency it was measured over: one element per host layer
static llama_tenants tenants_for(const llama_residency & r) {
    llama_tenants out;
    for (size_t i = 0; i < r.size(); i++) {
        if (r[i]) {
            ggml_sycl_context_tenant_desc d = {};
            d.struct_size                   = sizeof(d);
            d.cohort                        = GGML_SYCL_CONTEXT_COHORT_COMPUTE;
            d.slot_index                    = (uint32_t) i;
            d.device                        = 0;
            d.slot_bytes                    = 1000 + i;
            out.push_back(d);
        }
    }
    return out;
}

// the host layers a tenant section was measured over, read back
static llama_residency residency_of(const llama_tenants & t, size_t n_layer) {
    llama_residency r(n_layer, 0);
    for (const auto & d : t) {
        r[d.slot_index] = 1;
    }
    return r;
}

struct stub {
    size_t                                                n_layer = 0;
    std::function<llama_residency(const llama_tenants *)> probe_fn;
    std::vector<llama_residency>                          measured_over;
    int                                                   probe_calls = 0;
    bool                                                  measure_ok  = true;
};

static llama_residency_fixpoint_result run(stub & s) {
    return llama_residency_fixpoint(
        s.n_layer,
        [&](const llama_tenants * t) {
            s.probe_calls++;
            return s.probe_fn(t);
        },
        [&](const llama_residency & r, llama_tenants & out) {
            s.measured_over.push_back(r);
            if (!s.measure_ok) {
                return false;
            }
            out = tenants_for(r);
            return true;
        });
}

static void test_decision() {
    // (1)
    for (bool plan_active : { false, true }) {
        CHECK(llama_plan_caps_decide(false, plan_active, true, true, true) == LLAMA_PLAN_CAPS_UNPLANNED);
    }
    for (bool l4 : { false, true }) {
        CHECK(llama_plan_caps_decide(true, false, true, true, l4) == LLAMA_PLAN_CAPS_UNPLANNED);
        CHECK(llama_plan_caps_decide(true, false, false, false, l4) == LLAMA_PLAN_CAPS_UNPLANNED);
    }
    // (2) both directions of the ordering: L4 absent gives UNPLANNED with the cap procs absent (never
    // REFUSE) and present; L4 present with the cap procs absent gives REFUSE (case 3)
    CHECK(llama_plan_caps_decide(true, true, true, true, false) == LLAMA_PLAN_CAPS_UNPLANNED);
    CHECK(llama_plan_caps_decide(true, true, false, false, false) == LLAMA_PLAN_CAPS_UNPLANNED);
    CHECK(llama_plan_caps_decide(true, true, false, true, false) == LLAMA_PLAN_CAPS_UNPLANNED);
    CHECK(llama_plan_caps_decide(true, true, true, false, false) == LLAMA_PLAN_CAPS_UNPLANNED);
    // (3)
    CHECK(llama_plan_caps_decide(true, true, true, true, true) == LLAMA_PLAN_CAPS_ACQUIRE);
    CHECK(llama_plan_caps_decide(true, true, false, true, true) == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS);
    CHECK(llama_plan_caps_decide(true, true, true, false, true) == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS);
    CHECK(llama_plan_caps_decide(true, true, false, false, true) == LLAMA_PLAN_CAPS_REFUSE_NO_PROCS);
    CHECK(std::string(llama_plan_caps_missing_procs_reason()) ==
          "SYCL placement plan active but ggml_backend_sycl_plan_caps_new/_free is not exported");
}

static void test_no_demotion() {
    // (4)
    stub s;
    s.n_layer  = 4;
    s.probe_fn = [](const llama_tenants *) {
        return llama_residency(4, 0);
    };
    const auto r = run(s);
    CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
    CHECK(r.iterations == 1);
    REQUIRE(s.measured_over.size() == 1);
    CHECK(r.residency == llama_residency(4, 0));
    CHECK(r.tenants.empty());
    CHECK(!r.verify_shrunk);
}

static void test_one_demotion() {
    // (5): the first probe (no tenants) is clean; under the tenants measured over the clean residency
    // it demotes layer 1; under the tenants measured over {1} it demotes nothing more.
    stub s;
    s.n_layer  = 3;
    s.probe_fn = [](const llama_tenants * t) {
        llama_residency r(3, 0);
        if (t != nullptr && t->empty()) {
            r[1] = 1;
        } else if (t != nullptr) {
            r = residency_of(*t, 3);
        }
        return r;
    };
    const auto r = run(s);
    CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
    CHECK(r.iterations == 2);
    REQUIRE(s.measured_over.size() == 2);
    CHECK(s.measured_over[0] == llama_residency(3, 0));
    CHECK((s.measured_over[1] == llama_residency{ 0, 1, 0 }));
    CHECK((r.residency == llama_residency{ 0, 1, 0 }));
    REQUIRE(r.tenants.size() == 1);
    CHECK(r.tenants[0].slot_index == 1);
}

static void test_demote_only() {
    // (6): R0 = {1,0}. Over tenants measured over {1,0} the probe answers {0,0} (a promotion); over
    // tenants measured over {0,0} it answers {1,0} again. The iteration must not follow the promotion:
    // R stays {1,0}, one MEASURE. (The verify then tries {0,0}, the probe does not keep it, and
    // {1,0} stands.)
    stub s;
    s.n_layer  = 2;
    s.probe_fn = [](const llama_tenants * t) {
        if (t == nullptr) {
            return llama_residency{ 1, 0 };
        }
        return residency_of(*t, 2)[0] ? llama_residency{ 0, 0 } : llama_residency{ 1, 0 };
    };
    const auto r = run(s);
    CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
    CHECK(r.iterations == 1);
    REQUIRE(!s.measured_over.empty());
    CHECK((s.measured_over[0] == llama_residency{ 1, 0 }));
    CHECK((r.residency == llama_residency{ 1, 0 }));
    CHECK(!r.verify_shrunk);
}

static void test_worst_chain() {
    // (7): each MEASURE exposes exactly one more host layer; n_layer demotions after R0 = none
    const size_t n_layer = 6;
    stub         s;
    s.n_layer  = n_layer;
    s.probe_fn = [n_layer](const llama_tenants * t) {
        llama_residency r(n_layer, 0);
        if (t != nullptr) {
            r        = residency_of(*t, n_layer);
            size_t n = 0;
            for (uint8_t b : r) {
                n += b;
            }
            if (n < n_layer) {
                r[n] = 1;  // one more host layer than the tenants were measured over
            }
        }
        return r;
    };
    const auto r = run(s);
    CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
    CHECK(r.iterations == n_layer + 1);
    CHECK(r.residency == llama_residency(n_layer, 1));
}

static void test_measure_failed() {
    // (8)
    stub s;
    s.n_layer    = 3;
    s.measure_ok = false;
    s.probe_fn   = [](const llama_tenants *) {
        return llama_residency(3, 0);
    };
    const auto r = run(s);
    CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_MEASURE_FAILED);
    CHECK(!r.reason.empty());
    CHECK(r.tenants.empty());
}

static void test_wrong_length() {
    // (9)
    stub s;
    s.n_layer  = 3;
    s.probe_fn = [](const llama_tenants *) {
        return llama_residency(2, 0);
    };
    CHECK(run(s).status == LLAMA_RESIDENCY_FIXPOINT_BUG);

    stub s2;
    s2.n_layer  = 3;
    s2.probe_fn = [](const llama_tenants * t) {
        return t == nullptr ? llama_residency(3, 0) : llama_residency(4, 0);
    };
    CHECK(run(s2).status == LLAMA_RESIDENCY_FIXPOINT_BUG);
}

// a probe that, over any tenants, answers `converged` during the iteration (it is the union with R)
// and `raw` when asked for the verify -- which the stub tells apart by the tenants it is handed: the
// section measured over R* (the verify's first call) and the section measured over R_v (its second).
static void test_verify() {
    // (10a): R* = {0,1,1}; the raw probe over T* answers {0,1,0}; over T_v (measured over {0,1,0})
    // it keeps {0,1,0}: the pair is accepted.
    {
        stub s;
        s.n_layer  = 3;
        s.probe_fn = [](const llama_tenants * t) {
            if (t == nullptr) {
                return llama_residency{ 0, 1, 1 };  // R0
            }
            const llama_residency over = residency_of(*t, 3);
            if (over == llama_residency{ 0, 1, 1 }) {
                return llama_residency{ 0, 1, 0 };  // raw: the third layer would stay on device
            }
            return over;                            // keeps whatever it was measured over
        };
        const auto r = run(s);
        CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
        CHECK(r.verify_shrunk);
        CHECK((r.residency == llama_residency{ 0, 1, 0 }));
        CHECK((residency_of(r.tenants, 3) == llama_residency{ 0, 1, 0 }));
        REQUIRE(s.measured_over.size() >= 2);
        CHECK((s.measured_over.back() == llama_residency{ 0, 1, 0 }));
    }
    // (10b): the probe over T_v does not keep R_v (it asks for the third layer back on host):
    // (R*, T*) stands
    {
        stub s;
        s.n_layer  = 3;
        s.probe_fn = [](const llama_tenants * t) {
            if (t == nullptr) {
                return llama_residency{ 0, 1, 1 };
            }
            const llama_residency over = residency_of(*t, 3);
            if (over == llama_residency{ 0, 1, 1 }) {
                return llama_residency{ 0, 1, 0 };
            }
            return llama_residency{ 0, 1, 1 };
        };
        const auto r = run(s);
        CHECK(r.status == LLAMA_RESIDENCY_FIXPOINT_OK);
        CHECK(!r.verify_shrunk);
        CHECK((r.residency == llama_residency{ 0, 1, 1 }));
        CHECK((residency_of(r.tenants, 3) == llama_residency{ 0, 1, 1 }));
    }
}

static void test_verify_bug() {
    // (11): R* = {0,0,0}; the raw probe over T* holds layer 2, which R* does not: R* was no fixpoint.
    // Only the verify can reach this (the iteration unions, so it cannot), so the stub tells the
    // verify from the iteration by the call count.
    stub s;
    s.n_layer  = 3;
    int calls  = 0;
    s.probe_fn = [&calls](const llama_tenants * t) {
        calls++;
        if (t == nullptr) {
            return llama_residency(3, 0);
        }
        // the iteration's probe over T0 (call 2) is clean; the verify's raw probe (call 3) is not
        return calls == 2 ? llama_residency(3, 0) : llama_residency{ 0, 0, 1 };
    };
    CHECK(run(s).status == LLAMA_RESIDENCY_FIXPOINT_BUG);
}

int main() {
    test_decision();
    test_no_demotion();
    test_one_demotion();
    test_demote_only();
    test_worst_chain();
    test_measure_failed();
    test_wrong_length();
    test_verify();
    test_verify_bug();
    if (n_failed != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", n_failed);
        return 1;
    }
    std::printf("test-residency-fixpoint: all cases passed\n");
    return 0;
}
