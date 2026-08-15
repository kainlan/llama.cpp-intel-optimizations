// Host-only gate for the batched-MoE MUL_MAT_ID coverage tables
// (ggml/src/ggml-sycl/moe-mmvq-tables.hpp).
//
// No SYCL, no oneAPI, no GPU: the tables are pure functions of
// (ggml_type, ggml_layout_mode), so any agent can run this.
//
// What it protects (llama.cpp-gx30): the capability query may only advertise
// (type, layout) pairs that some executor can actually run. When capability
// over-advertises, the route is admitted, the executor declines at submit, and
// the decline escapes as ggml_sycl_fallback_error -> GGML_STATUS_FAILED --
// which ggml_backend_compare_graph_backend discards, so test-backend-ops
// compares an unwritten dst and reports a numeric error instead of a refusal.
// Adding a type to one table without the other must be caught here, not there.

#include "../ggml/src/ggml-sycl/moe-mmvq-tables.hpp"
#include "ggml.h"

#include <cstdio>
#include <vector>

namespace {

const std::vector<ggml_layout_mode> & all_layouts() {
    // ggml_layout_mode has no COUNT sentinel, so the list is explicit. The
    // assertion below catches an insertion into the middle of the enum. A
    // layout APPENDED after the current last one is not caught, but it defaults
    // to unsupported in both tables, so the subset invariant still holds for it.
    static const std::vector<ggml_layout_mode> layouts = {
        GGML_LAYOUT_AOS,
        GGML_LAYOUT_SOA,
        GGML_LAYOUT_COALESCED,
        GGML_LAYOUT_MXFP4_I8,
        GGML_LAYOUT_XMX_TILED,
        GGML_LAYOUT_XMX_TILED_BUNDLE4,
        GGML_LAYOUT_XMX_GEMM_TILED,
        GGML_LAYOUT_ONEDNN_PACKED,
        GGML_LAYOUT_ONEDNN_WOQ,
        GGML_LAYOUT_MXFP4_DPAS,
    };
    return layouts;
}

using layout_predicate = bool (*)(ggml_type, ggml_layout_mode);

// The subset check itself, factored out so the positive control below can drive
// the exact same code path with data that is known to violate the invariant.
// A checker that has never been observed to fail is not evidence.
int count_over_advertised(layout_predicate capability, layout_predicate dispatch, bool report) {
    int violations = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = static_cast<ggml_type>(t);
        for (const ggml_layout_mode layout : all_layouts()) {
            if (capability(type, layout) && !dispatch(type, layout)) {
                ++violations;
                if (report) {
                    std::printf(
                        "  FAIL over-advertised: type=%d layout=%d admitted by capability, "
                        "refused by every executor\n",
                        t, static_cast<int>(layout));
                }
            }
        }
    }
    return violations;
}

bool cap_wrapper(ggml_type t, ggml_layout_mode l) {
    return moe_mmvq_capability_supports_layout(t, l);
}

bool dispatch_wrapper(ggml_type t, ggml_layout_mode l) {
    return moe_mmvq_any_dispatch_supports_layout(t, l);
}

// Deliberately inconsistent pair for the positive control: capability admits
// Q4_K/AOS, no executor does.
bool fake_cap_over_advertising(ggml_type t, ggml_layout_mode l) {
    return cap_wrapper(t, l) || (t == GGML_TYPE_Q4_K && l == GGML_LAYOUT_AOS);
}

}  // namespace

int main() {
    int failures = 0;

    // 1. Enum shape: catches a layout inserted mid-enum, which would silently
    //    renumber the values this test enumerates.
    if (static_cast<int>(GGML_LAYOUT_MXFP4_DPAS) != 9 || all_layouts().size() != 10) {
        std::printf(
            "FAIL: ggml_layout_mode changed shape (MXFP4_DPAS=%d, listed=%zu); "
            "update all_layouts()\n",
            static_cast<int>(GGML_LAYOUT_MXFP4_DPAS), all_layouts().size());
        ++failures;
    }

    // 2. Positive control FIRST -- a control placed after the checks it protects
    //    explains a failure but cannot prevent one.
    const int control_violations = count_over_advertised(fake_cap_over_advertising, dispatch_wrapper, false);
    if (control_violations != 1) {
        std::printf(
            "FAIL: positive control expected exactly 1 detected violation, got %d. "
            "The subset checker is not sensitive and its zero below would mean nothing.\n",
            control_violations);
        ++failures;
    }

    // 3. The invariant: capability must be a subset of the executor union.
    const int violations = count_over_advertised(cap_wrapper, dispatch_wrapper, true);
    if (violations != 0) {
        std::printf("FAIL: %d capability/dispatch pair(s) over-advertised\n", violations);
        ++failures;
    }

    // 4. Anchors that name the coverage this ticket added, so a regression that
    //    re-shadows the Q1_0/NVFP4 AoS route is named rather than merely counted.
    const struct {
        ggml_type        type;
        ggml_layout_mode layout;
        const char *     name;
    } required[] = {
        { GGML_TYPE_Q1_0,  GGML_LAYOUT_AOS, "q1_0/AOS"  },
        { GGML_TYPE_NVFP4, GGML_LAYOUT_AOS, "nvfp4/AOS" },
        { GGML_TYPE_Q4_0,  GGML_LAYOUT_AOS, "q4_0/AOS"  },
        { GGML_TYPE_Q8_0,  GGML_LAYOUT_AOS, "q8_0/AOS"  },
    };

    for (const auto & req : required) {
        if (!moe_mmvq_batched_dispatch_supports_layout(req.type, req.layout)) {
            std::printf("FAIL: %s must be executable by the batched dispatch\n", req.name);
            ++failures;
        }
        if (!moe_mmvq_capability_supports_layout(req.type, req.layout)) {
            std::printf("FAIL: %s must be advertisable by the capability query\n", req.name);
            ++failures;
        }
    }

    // 5. The types with no _id kernel family must stay unadvertised. This is the
    //    half that keeps a future coverage pass honest: widening capability for
    //    these without adding the kernel turns a clean refusal into a wrong answer.
    const ggml_type uncovered[] = { GGML_TYPE_Q4_K,   GGML_TYPE_Q6_K, GGML_TYPE_Q4_1,
                                    GGML_TYPE_IQ4_XS, GGML_TYPE_F16,  GGML_TYPE_F32 };
    for (const ggml_type type : uncovered) {
        for (const ggml_layout_mode layout : all_layouts()) {
            if (moe_mmvq_capability_supports_layout(type, layout)) {
                std::printf("FAIL: type=%d layout=%d advertised but has no _id kernel family\n", static_cast<int>(type),
                            static_cast<int>(layout));
                ++failures;
            }
        }
    }

    if (failures != 0) {
        std::printf("test-sycl-moe-mmvq-tables: FAILED (%d)\n", failures);
        return 1;
    }
    std::printf("test-sycl-moe-mmvq-tables: OK\n");
    return 0;
}
