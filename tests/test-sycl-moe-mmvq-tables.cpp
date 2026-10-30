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
#include <cstring>
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

// The pair the positive control injects. It must be one no executor covers, or
// the control silently detects nothing and stops being a control.
//
// This bit rots: the control originally injected Q4_K/AOS, and when wave 1 gave
// Q4_K a real _id kernel the injection became legitimate and the control went
// void -- reporting 0 violations while claiming to prove sensitivity. Q4_0 with
// a oneDNN packed layout is durable instead: Q4_0 is covered for AoS only, and
// the MMVQ path has no packed-layout kernel for any type. control_premise_holds()
// below asserts that, so if a future wave ever covers it the control fails loudly
// rather than quietly passing.
constexpr ggml_type        k_control_type   = GGML_TYPE_Q4_0;
constexpr ggml_layout_mode k_control_layout = GGML_LAYOUT_ONEDNN_WOQ;

bool control_premise_holds() {
    return !moe_mmvq_any_dispatch_supports_layout(k_control_type, k_control_layout);
}

bool fake_cap_over_advertising(ggml_type t, ggml_layout_mode l) {
    return cap_wrapper(t, l) || (t == k_control_type && l == k_control_layout);
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
    //    explains a failure but cannot prevent one. Its premise is checked before
    //    the control itself, because an injection that is no longer a violation
    //    makes the control pass vacuously.
    if (!control_premise_holds()) {
        std::printf(
            "FAIL: positive control premise broken -- type=%d layout=%d is now covered by an "
            "executor, so injecting it no longer creates a violation. Pick an uncovered pair.\n",
            static_cast<int>(k_control_type), static_cast<int>(k_control_layout));
        ++failures;
    }
    //    Measured as a DELTA against the real tables, not as an absolute count:
    //    on a tree that already has a genuine violation an absolute "== 1" fails
    //    and blames the checker for being insensitive, which is the opposite of
    //    what is happening. The delta isolates the injection.
    const int real_violations    = count_over_advertised(cap_wrapper, dispatch_wrapper, false);
    const int control_violations = count_over_advertised(fake_cap_over_advertising, dispatch_wrapper, false);
    if (control_violations - real_violations != 1) {
        std::printf(
            "FAIL: positive control did not detect its injected violation "
            "(real=%d, with-injection=%d, delta=%d, want 1). "
            "The subset checker is not sensitive, so its result below would mean nothing.\n",
            real_violations, control_violations, control_violations - real_violations);
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
        // gx30: every type with a generic mul_mat_vec_q<> tuple to transcribe
        { GGML_TYPE_Q4_1,  GGML_LAYOUT_AOS, "q4_1/AOS"  },
        { GGML_TYPE_Q4_K,  GGML_LAYOUT_AOS, "q4_K/AOS"  },
        { GGML_TYPE_Q5_K,  GGML_LAYOUT_AOS, "q5_K/AOS"  },
        { GGML_TYPE_Q6_K,  GGML_LAYOUT_AOS, "q6_K/AOS"  },
        { GGML_TYPE_Q5_0,  GGML_LAYOUT_AOS, "q5_0/AOS"  },
        { GGML_TYPE_Q5_1,  GGML_LAYOUT_AOS, "q5_1/AOS"  },
        { GGML_TYPE_Q2_K,  GGML_LAYOUT_AOS, "q2_K/AOS"  },
        { GGML_TYPE_Q3_K,  GGML_LAYOUT_AOS, "q3_K/AOS"  },
        // s36q: the first iq* type. IQ4_NL is Q4_0-shaped (32-wide blocks, qi=4, vdr=2) and its dense MMVQ
        // kernel is the generic mul_mat_vec_q<> body with vec_dot_iq4_nl_q8_1, so the generic _id path
        // takes it with that vec_dot and no new kernel body.
        { GGML_TYPE_IQ4_NL, GGML_LAYOUT_AOS, "iq4_nl/AOS" },
        // s36q phase 2: IQ3_XXS and IQ3_S. Their dense kernels pass grid tables to a vec_dot with a
        // non-generic signature, so the _id path wraps each in a generic-signature adaptor over the
        // same grid tables the dense kernels read (qi = QI3_x / 2, vdr = 1, as the dense kernels use).
        { GGML_TYPE_IQ3_XXS, GGML_LAYOUT_AOS, "iq3_xxs/AOS" },
        { GGML_TYPE_IQ3_S,   GGML_LAYOUT_AOS, "iq3_s/AOS"   },
        // s36q phase 3: IQ2_XXS, IQ2_XS and IQ2_S, same (QK_K, QI2_x / 2, vdr 1) dense tuple. The first two
        // take grid tables through adaptors; IQ2_S's vec_dot already has the generic signature.
        { GGML_TYPE_IQ2_XXS, GGML_LAYOUT_AOS, "iq2_xxs/AOS" },
        { GGML_TYPE_IQ2_XS,  GGML_LAYOUT_AOS, "iq2_xs/AOS"  },
        { GGML_TYPE_IQ2_S,   GGML_LAYOUT_AOS, "iq2_s/AOS"   },
        // s36q phase 4: Q2_0 (64-wide blocks, qi = QK2_0 / 32 q8_1 chunks, vdr 1, the Q1_0 tuple shape) with
        // a signature-generic vec_dot, so the generic _id path takes it with no adaptor. Until phase 4 the
        // type had no SYCL kernel of any kind and was refused (yitq/phbr).
        { GGML_TYPE_Q2_0,    GGML_LAYOUT_AOS, "q2_0/AOS"    },
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
    // The two families that remain refused, for different reasons. IQ4_XS, IQ1_S and
    // IQ1_M are refused only because no _id tuple and, where their vec_dot takes grid
    // tables, no generic-signature adaptor has been written for them yet (IQ4_NL, the
    // IQ3 and the IQ2 types are covered since s36q); each moves out of this list in
    // the change that adds its kernel. The float types cannot use MMVQ at all -- it
    // quantizes the activation to Q8_1 and dispatches vec_dot_*_q8_1, meaningless for
    // float weights. Covering either must move this list in the same change, which
    // is the point.
    const ggml_type uncovered[] = { GGML_TYPE_IQ4_XS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_F16,
                                    GGML_TYPE_F32,    GGML_TYPE_BF16 };
    for (const ggml_type type : uncovered) {
        for (const ggml_layout_mode layout : all_layouts()) {
            if (moe_mmvq_capability_supports_layout(type, layout)) {
                std::printf("FAIL: type=%d layout=%d advertised but has no _id kernel family\n", static_cast<int>(type),
                            static_cast<int>(layout));
                ++failures;
            }
        }
    }

    // 6. MUL_MAT_ID ADMISSION population (llama.cpp-yitq).
    //    ggml_backend_sycl_device_supports_op's MUL_MAT_ID branch used to gate on
    //    ggml_sycl_mul_mat_type_supported() -- the DENSE MUL_MAT allowlist. That
    //    allowlist is legitimately true for F32/F16/IQ1_S..IQ4_XS (they have real
    //    dense kernels) and none of them has an _id kernel family, so admission
    //    handed the scheduler 234 census cases this backend cannot compute. The
    //    route oracle refused them, but per the header comment that refusal is
    //    discarded and resurfaces as ERR 86-99 instead of a CPU fallback.
    //
    //    Pinned as an exact PARTITION of GGML_TYPE_COUNT, not as two spot lists:
    //    a spot check passes while a newly added type silently joins neither set.
    //    The cardinality assertion below is what makes the lists load-bearing --
    //    identity plus population, because either alone fails open.
    const ggml_type admission_expected[] = {
        GGML_TYPE_Q1_0,  GGML_TYPE_NVFP4,   GGML_TYPE_Q4_0,   GGML_TYPE_Q8_0,   GGML_TYPE_MXFP4,
        GGML_TYPE_Q4_1,  GGML_TYPE_Q4_K,    GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K,   GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,  GGML_TYPE_Q2_K,    GGML_TYPE_Q3_K,   GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,  GGML_TYPE_Q2_0,
    };
    //    The exact set that regressed: dense MUL_MAT kernels exist, _id does not.
    //    BF16/tq2_0 are absent on purpose -- 186348705 already refuses them by
    //    leaving them out of the dense allowlist, so they never reached here. q2_0
    //    was refused the same way until s36q phase 4 gave it both kernels.
    const ggml_type admission_refused[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_XS,
    };

    for (const ggml_type type : admission_expected) {
        if (!moe_mmvq_admission_supports_type(type)) {
            std::printf("FAIL: type=%d has _id coverage but MUL_MAT_ID admission refuses it\n", static_cast<int>(type));
            ++failures;
        }
    }
    for (const ggml_type type : admission_refused) {
        if (moe_mmvq_admission_supports_type(type)) {
            std::printf(
                "FAIL: type=%d admitted for MUL_MAT_ID with no _id kernel family -- this is the "
                "yitq regression (wrong numbers instead of CPU fallback)\n",
                static_cast<int>(type));
            ++failures;
        }
    }
    //    Cardinality: nothing outside admission_expected may be admitted. Catches
    //    a type added to the dispatch tables without being added to the list here.
    int admitted_total = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        if (moe_mmvq_admission_supports_type(static_cast<ggml_type>(t))) {
            ++admitted_total;
        }
    }
    const int admission_expected_count = static_cast<int>(sizeof(admission_expected) / sizeof(admission_expected[0]));
    if (admitted_total != admission_expected_count) {
        std::printf(
            "FAIL: MUL_MAT_ID admission population is %d, expected %d. A type gained (or lost) _id "
            "coverage without this list moving in the same change.\n",
            admitted_total, admission_expected_count);
        ++failures;
    }
    //    Sensitivity, stated honestly. Measured by mutating the predicate
    //    (2026-09-17): admitting F32, admitting an IQ type, dropping a covered
    //    type, and admitting everything are all CAUGHT by the checks above.
    //    Dropping the predicate's capability axis entirely is NOT caught, and
    //    cannot be, because moe_mmvq_batched_dispatch_supports_type and the
    //    capability table cover exactly the same 20 types today -- so the two
    //    axes are indistinguishable by population. The second axis is therefore
    //    defensive, not gated here; what keeps the sets coinciding is the subset
    //    invariant in section 3. Do not add a control that "proves" the axis by
    //    reimplementing the predicate over injected tables: that would exercise
    //    the replica, not the function, and pass while the real one is broken.
    //    If a future wave ever gives a type an _id executor without advertising
    //    a layout for it, the axis becomes observable -- add the case then.
    //
    //    What IS checkable now is the premise that makes the two axes agree:
    //    every admitted type must be advertised for at least one layout. If this
    //    fails, the sets have diverged and the note above is stale.
    for (const ggml_type type : admission_expected) {
        bool advertised = false;
        for (const ggml_layout_mode layout : all_layouts()) {
            if (moe_mmvq_capability_supports_layout(type, layout)) {
                advertised = true;
                break;
            }
        }
        if (!advertised) {
            std::printf(
                "FAIL: type=%d is admitted for MUL_MAT_ID but advertised for no layout -- the two "
                "admission axes have diverged; the capability axis is now load-bearing and needs "
                "its own case here\n",
                static_cast<int>(type));
            ++failures;
        }
    }

    // 7. The grouped XMX_TILED executor must accept PARTIAL cover (llama.cpp-4hg7).
    //    A hybrid decode routes some slots to the device and the rest to the host. The
    //    kernel is driven by per-slot route arrays, so a partial cover is as valid as a
    //    full one provided those arrays exist. Declining it dropped the op onto a
    //    per-expert fallback that read xmx_tiled bytes as AOS.
    const struct {
        bool         full;
        int          n_gpu;
        bool         arrays;
        bool         want;
        const char * name;
    } cover_cases[] = {
        { true,  4, true,  true,  "full cover with arrays"                           },
        { true,  4, false, true,  "full cover, device-built groups (no host arrays)" },
        { false, 1, true,  true,  "partial cover, one device slot of four (4hg7)"    },
        { false, 3, true,  true,  "partial cover, three device slots of four"        },
        { false, 1, false, false, "partial cover without route arrays is refused"    },
        { false, 0, true,  false, "no device entries at all"                         },
        { true,  0, true,  false, "full cover claimed over zero entries"             },
    };

    for (const auto & c : cover_cases) {
        if (moe_mmvq_xmx_tiled_grouped_accepts_cover(c.full, c.n_gpu, c.arrays) != c.want) {
            std::printf("FAIL: xmx_tiled grouped cover policy wrong for: %s (want %d)\n", c.name, c.want ? 1 : 0);
            ++failures;
        }
    }

    // 8. The per-expert MXFP4 direct path reads exactly AOS/SOA/COALESCED. Every other
    //    layout in the enum is a different byte format and must be refused there, never
    //    mapped onto AOS (llama.cpp-4hg7). Pinned over the whole layout list so a layout
    //    added to the direct path without a decoder turns this red.
    for (const ggml_layout_mode layout : all_layouts()) {
        const bool expect = layout == GGML_LAYOUT_AOS || layout == GGML_LAYOUT_SOA || layout == GGML_LAYOUT_COALESCED;
        if (moe_mmvq_mxfp4_direct_reads_layout(layout) != expect) {
            std::printf("FAIL: mxfp4 direct path layout readability wrong for layout=%d (want %d)\n",
                        static_cast<int>(layout), expect ? 1 : 0);
            ++failures;
        }
    }
    //    The layout this ticket was about must be among the refused ones, and the batched
    //    executor must still be the thing that reads it.
    if (moe_mmvq_mxfp4_direct_reads_layout(GGML_LAYOUT_XMX_TILED) ||
        !moe_mmvq_batched_dispatch_supports_layout(GGML_TYPE_MXFP4, GGML_LAYOUT_XMX_TILED)) {
        std::printf("FAIL: XMX_TILED must be batched-executable and not readable by the direct path\n");
        ++failures;
    }

    // 9. A prompt-phase layout over a MIXED tensor is executable (llama.cpp-f6zo).
    //    GPT-OSS at PCT=60 keeps blk.12 as 12 device experts (xmx_tiled) and 20 host
    //    experts (host AOS, executed on the CPU). The prompt probe found local=12 host=20
    //    missing=0 for xmx_tiled and still aborted, because admission demanded
    //    local == n_experts && host == 0 -- "all experts on the device" -- and so asked for
    //    a SOA copy the planner never builds. Placement decides the executor: device entries
    //    at their loaded layout plus host entries on the CPU cover every expert.
    //    Host-only (local == 0) stays with the SOA/host path; a missing or secondary expert,
    //    or a cover that does not add up to the tensor, is not executable.
    const struct {
        size_t       local, secondary, host, missing, n_experts;
        bool         want;
        const char * name;
    } prompt_cover_cases[] = {
        { 12, 0, 20, 0, 32, true,  "mixed tensor, 12 device + 20 host (f6zo)"                 },
        { 1,  0, 31, 0, 32, true,  "mixed tensor, one device expert"                          },
        { 31, 0, 1,  0, 32, true,  "mixed tensor, one host expert"                            },
        { 32, 0, 0,  0, 32, true,  "all device (the case the strict check always accepted)"   },
        { 0,  0, 32, 0, 32, false, "all host is not a device-layout cover"                    },
        { 0,  0, 0,  1, 32, false, "SOA asked of xmx_tiled entries: local=0 host=0 missing=1" },
        { 12, 0, 19, 1, 32, false, "one expert missing"                                       },
        { 12, 0, 20, 1, 32, false, "an expert counted missing on top of a full cover"         },
        { 12, 1, 20, 0, 32, false, "a secondary expert counted on top of a full cover"        },
        { 12, 0, 18, 0, 32, false, "cover does not add up to the tensor"                      },
        { 11, 1, 20, 0, 32, false, "a secondary-device expert (PP is unvalidated there)"      },
        { 12, 0, 20, 0, 0,  false, "empty tensor"                                             },
    };

    for (const auto & c : prompt_cover_cases) {
        if (moe_mmvq_prompt_layout_cover_executable(c.local, c.secondary, c.host, c.missing, c.n_experts) != c.want) {
            std::printf("FAIL: prompt layout cover policy wrong for: %s (want %d)\n", c.name, c.want ? 1 : 0);
            ++failures;
        }
    }

    // 7. AoS expert-handle publication (llama.cpp-s36q, test-backend-ops case for issue 27873:
    //    MUL_MAT_ID with n_mats = 1, whose src0 is ne[2] == 1 and unclassified by name).
    //    At upload the buffer cannot tell such a tensor from a dense 2D weight, so it must not
    //    publish; a MUL_MAT_ID dispatch knows its src0 is an expert tensor and must publish, or
    //    the retained resolver finds no expert and the op fails (NOT_FOUND).
    const struct {
        bool        classified;
        int64_t     ne2;
        bool        mul_mat_id;
        bool        want;
        const char * name;
    } publication_cases[] = {
        { false, 1, false, false, "dense 2D weight at upload"                       },
        { false, 1, true,  true,  "single-expert src0 reached by a MUL_MAT_ID"      },
        { false, 4, false, true,  "structural expert tensor at upload"              },
        { false, 4, true,  true,  "structural expert tensor at dispatch"            },
        { true,  1, false, true,  "name-classified single-expert tensor at upload"  },
        { true,  1, true,  true,  "name-classified single-expert tensor at dispatch" },
    };
    for (const auto & c : publication_cases) {
        if (moe_aos_expert_publication_wanted(c.classified, c.ne2, c.mul_mat_id) != c.want) {
            std::printf("FAIL: AoS expert publication wrong for: %s (want %d)\n", c.name, c.want ? 1 : 0);
            ++failures;
        }
    }

    // 8. The batched mul_mat_id prompt executor's tables (llama.cpp-348c). The f16 arm admits a (type, layout) pair
    //    when a kernel decodes one expert of that materialized layout to f16: every quantized type
    //    ggml_get_to_fp16_sycl_for_layout decodes, on AOS; the four with a per-expert SOA dequant (Q4_0, Q8_0, Q4_K,
    //    Q6_K) on SOA too, since the planner materializes experts SOA (the B50 does Q8_0 and Q6_K); MXFP4 on SOA
    //    only. Q1_0 and NVFP4 are left out, because only their direct kernel may run them on a device. The list is
    //    the whole set: any other type the f16 arm admits on AOS fails below.
    const ggml_type pp_f16_aos_types[] = {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q2_0,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ1_M,
        // Not run by test-moe-pp-batched-dequant: they need an importance matrix to quantize.
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
    };
    for (const ggml_type type : pp_f16_aos_types) {
        const bool soa_too =
            type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K;
        for (const ggml_layout_mode layout : all_layouts()) {
            const bool want = layout == GGML_LAYOUT_AOS || (soa_too && layout == GGML_LAYOUT_SOA);
            if (moe_pp_batched_f16_supports_layout(type, layout) != want) {
                std::printf("FAIL: batched f16 arm wrong for type=%s layout=%d (want %d)\n", ggml_type_name(type),
                            static_cast<int>(layout), want ? 1 : 0);
                ++failures;
            }
        }
    }
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type   = static_cast<ggml_type>(t);
        bool            listed = type == GGML_TYPE_MXFP4;
        for (const ggml_type listed_type : pp_f16_aos_types) {
            listed = listed || listed_type == type;
        }
        if (!listed && moe_pp_batched_f16_supports_layout(type, GGML_LAYOUT_AOS)) {
            std::printf("FAIL: batched f16 arm admits type=%d on AOS, which the table above does not list\n", t);
            ++failures;
        }
    }
    const ggml_type pp_never_types[] = { GGML_TYPE_Q1_0, GGML_TYPE_NVFP4, GGML_TYPE_F16, GGML_TYPE_F32,
                                         GGML_TYPE_BF16 };
    for (const ggml_type type : pp_never_types) {
        for (const ggml_layout_mode layout : all_layouts()) {
            if (moe_pp_batched_supports_layout(type, layout, true, true)) {
                std::printf("FAIL: batched executor admits type=%s layout=%d, a direct-kernel or float type\n",
                            ggml_type_name(type), static_cast<int>(layout));
                ++failures;
            }
        }
    }

    const struct {
        ggml_type        type;
        ggml_layout_mode layout;
        bool             woq_enabled;
        bool             aos_enabled;
        bool             want;
        const char *     name;
    } pp_batched_cases[] = {
        { GGML_TYPE_MXFP4,  GGML_LAYOUT_SOA,       false, false, true,  "MXFP4 SOA f16 arm, both knobs off"  },
        { GGML_TYPE_MXFP4,  GGML_LAYOUT_AOS,       true,  true,  false, "MXFP4 AOS has no arm"               },
        { GGML_TYPE_MXFP4,  GGML_LAYOUT_XMX_TILED, true,  false, true,  "MXFP4 XMX_TILED WOQ arm"            },
        { GGML_TYPE_MXFP4,  GGML_LAYOUT_XMX_TILED, false, true,  false, "MXFP4 XMX_TILED without WOQ"        },
        { GGML_TYPE_Q4_0,   GGML_LAYOUT_AOS,       false, true,  true,  "Q4_0 AOS f16 arm"                   },
        { GGML_TYPE_Q4_0,   GGML_LAYOUT_AOS,       true,  false, false, "Q4_0 AOS with the AOS knob off"     },
        { GGML_TYPE_Q4_0,   GGML_LAYOUT_SOA,       true,  true,  true,  "Q4_0 SOA f16 arm (reorder dequant)" },
        { GGML_TYPE_Q8_0,   GGML_LAYOUT_SOA,       true,  true,  true,  "Q8_0 SOA f16 arm (SOA row-major)"   },
        { GGML_TYPE_Q6_K,   GGML_LAYOUT_SOA,       true,  true,  true,  "Q6_K SOA f16 arm (reorder dequant)" },
        { GGML_TYPE_Q6_K,   GGML_LAYOUT_SOA,       true,  false, false, "Q6_K SOA with the AOS knob off"     },
        { GGML_TYPE_Q8_0,   GGML_LAYOUT_COALESCED, true,  true,  false, "Q8_0 COALESCED has no arm"          },
        { GGML_TYPE_IQ4_XS, GGML_LAYOUT_AOS,       false, true,  true,  "IQ4_XS AOS f16 arm"                 },
        { GGML_TYPE_IQ4_XS, GGML_LAYOUT_SOA,       false, true,  false, "IQ4_XS has no SOA dequant"          },
    };

    for (const auto & c : pp_batched_cases) {
        if (moe_pp_batched_supports_layout(c.type, c.layout, c.woq_enabled, c.aos_enabled) != c.want) {
            std::printf("FAIL: batched executor wrong for: %s (want %d)\n", c.name, c.want ? 1 : 0);
            ++failures;
        }
    }
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const ggml_type type = static_cast<ggml_type>(t);
        if (moe_pp_batched_woq_supports_layout(type, GGML_LAYOUT_SOA) && type != GGML_TYPE_MXFP4) {
            std::printf("FAIL: the WOQ arm admits type=%d; it is MXFP4 only\n", t);
            ++failures;
        }
        for (const ggml_layout_mode layout : all_layouts()) {
            // GGML_SYCL_MOE_PP_F16_BATCHED_AOS=0 must leave exactly the MXFP4 executor.
            if (moe_pp_batched_supports_layout(type, layout, true, false) && type != GGML_TYPE_MXFP4) {
                std::printf("FAIL: the AOS knob off still admits type=%d layout=%d\n", t, static_cast<int>(layout));
                ++failures;
            }
        }
    }

    // 9. Refusal names: every cause below COUNT has its own name, and the per-device latch's 32-bit mask holds
    //    them all.
    const int pp_refusal_count = static_cast<int>(moe_pp_batched_refusal::COUNT);
    if (pp_refusal_count > 32) {
        std::printf("FAIL: moe_pp_batched_refusal has %d causes; the refusal latch holds 32\n", pp_refusal_count);
        ++failures;
    }
    if (std::strcmp(moe_pp_batched_refusal_name(moe_pp_batched_refusal::COUNT), "unknown") != 0) {
        std::printf("FAIL: COUNT is named as if it were a cause\n");
        ++failures;
    }
    for (int a = 0; a < pp_refusal_count; ++a) {
        const char * name_a = moe_pp_batched_refusal_name(static_cast<moe_pp_batched_refusal>(a));
        if (std::strcmp(name_a, "unknown") == 0) {
            std::printf("FAIL: refusal %d has no name\n", a);
            ++failures;
        }
        for (int b = a + 1; b < pp_refusal_count; ++b) {
            if (std::strcmp(name_a, moe_pp_batched_refusal_name(static_cast<moe_pp_batched_refusal>(b))) == 0) {
                std::printf("FAIL: refusals %d and %d share the name %s\n", a, b, name_a);
                ++failures;
            }
        }
    }

    // 9b. Post-claim causes warn at default verbosity; prechecks stay behind the trace knob. The list is written out
    //     here rather than read from the header, so a cause moved across the split fails this table.
    const moe_pp_batched_refusal pp_post_claim[] = {
        moe_pp_batched_refusal::EXPERT_NOT_LOCAL,
        moe_pp_batched_refusal::EXPERT_LAYOUT,
        moe_pp_batched_refusal::XMX_TILE_CAPS,
        moe_pp_batched_refusal::F16_TILED,
        moe_pp_batched_refusal::NO_DEQUANT,
        moe_pp_batched_refusal::SCRATCH_SIZE,
        moe_pp_batched_refusal::SCRATCH_CACHE,
        moe_pp_batched_refusal::SCRATCH_RESERVE,
        moe_pp_batched_refusal::SCRATCH_SLOT,
        moe_pp_batched_refusal::SCRATCH_SLOT_SIZE,
        moe_pp_batched_refusal::GEMM,
    };
    for (int r = 0; r <= pp_refusal_count; ++r) {
        const moe_pp_batched_refusal refusal = static_cast<moe_pp_batched_refusal>(r);
        bool                         want    = false;
        for (const moe_pp_batched_refusal post : pp_post_claim) {
            want = want || post == refusal;
        }
        if (moe_pp_batched_refusal_after_claim(refusal) != want) {
            std::printf("FAIL: refusal %s is %s, want %s\n", moe_pp_batched_refusal_name(refusal),
                        want ? "trace-only" : "loud", want ? "loud (post-claim)" : "trace-only (precheck)");
            ++failures;
        }
    }

    // 10. Per-expert entries: the batched executor resolves each expert's own cache entry, so its claim takes a
    // planned layout only where those entries exist. A host-buffer weight (every default load) has one per expert in
    // any planned layout, staged by the load-end preload. A device-buffer weight (GGML_SYCL_WEIGHTS_EVICTABLE=0) has
    // them only on AOS, whose slices the upload publishes, because the upload reorders any other layout into one
    // block over every expert. MXFP4 keeps the route it had, so it is never declined here.
    const struct {
        ggml_type        type;
        ggml_layout_mode layout;
        bool             device_buffer;
        bool             want;
        const char *     name;
    } pp_entry_cases[] = {
        { GGML_TYPE_Q8_0,   GGML_LAYOUT_SOA, false, true,  "Q8_0 SOA in a host buffer (preloaded per expert)" },
        { GGML_TYPE_Q6_K,   GGML_LAYOUT_SOA, false, true,  "Q6_K SOA in a host buffer (preloaded per expert)" },
        { GGML_TYPE_Q8_0,   GGML_LAYOUT_SOA, true,  false, "Q8_0 SOA in a device buffer (one block)"          },
        { GGML_TYPE_Q6_K,   GGML_LAYOUT_SOA, true,  false, "Q6_K SOA in a device buffer (one block)"          },
        { GGML_TYPE_Q4_K,   GGML_LAYOUT_SOA, true,  false, "Q4_K SOA in a device buffer (one block)"          },
        { GGML_TYPE_Q6_K,   GGML_LAYOUT_AOS, true,  true,  "Q6_K AOS in a device buffer (published slices)"   },
        { GGML_TYPE_IQ4_XS, GGML_LAYOUT_AOS, true,  true,  "IQ4_XS AOS in a device buffer (published slices)" },
        { GGML_TYPE_MXFP4,  GGML_LAYOUT_SOA, true,  true,  "MXFP4 SOA in a device buffer (route unchanged)"   },
    };

    for (const auto & c : pp_entry_cases) {
        if (moe_pp_batched_has_expert_entries(c.type, c.layout, c.device_buffer) != c.want) {
            std::printf("FAIL: per-expert entries wrong for: %s (want %d)\n", c.name, c.want ? 1 : 0);
            ++failures;
        }
    }
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        for (const ggml_layout_mode layout : all_layouts()) {
            if (!moe_pp_batched_has_expert_entries(static_cast<ggml_type>(t), layout, false)) {
                std::printf("FAIL: a host-buffer weight is declined for type=%d layout=%d\n", t,
                            static_cast<int>(layout));
                ++failures;
            }
        }
    }

    // 11. The f16 arm's groups (llama.cpp-348c). The arm runs in the weight slot the plan sized for the ring and
    // groups the active experts to fit it: one group when every expert fits and no cap is smaller, otherwise groups
    // alternating between the slot's two halves, or reusing one buffer when the slot holds a single expert. The
    // device test's shape (32 experts of 768 x 2048) has a 835584-byte WOQ slot per expert, so its planned slot holds
    // 8 f16 experts of 3145728 bytes: 8 groups of 4 by default, 11 of 3 under the test's cap of 3.
    const size_t expert = size_t(768) * 2048 * 2;
    const size_t woq    = size_t(835584) * 32;

    const struct {
        size_t       slot, expert_bytes, active, cap;
        size_t       per, buffers, groups;
        const char * name;
    } group_cases[] = {
        { 8 * expert, expert, 8,  0, 8, 1, 1,  "every expert fits: one group, one buffer"         },
        { 8 * expert, expert, 8,  8, 8, 1, 1,  "a cap equal to the active count changes nothing"  },
        { 8 * expert, expert, 8,  3, 3, 2, 3,  "every expert fits but the cap splits them"        },
        { woq,        expert, 32, 0, 4, 2, 8,  "the device test's shape, no cap"                  },
        { woq,        expert, 32, 3, 3, 2, 11, "the device test's shape under its cap of 3"       },
        { woq,        expert, 32, 1, 1, 2, 32, "a cap of 1: one expert per group, two halves"     },
        { 3 * expert, expert, 4,  0, 1, 2, 4,  "a three-expert slot: two halves of one"           },
        { expert,     expert, 5,  0, 1, 1, 5,  "a one-expert slot: one buffer every group reuses" },
        { expert + 1, expert, 1,  0, 1, 1, 1,  "one active expert in a one-expert slot"           },
        { expert - 1, expert, 4,  0, 0, 0, 0,  "a slot smaller than one expert: no group"         },
        { 8 * expert, expert, 0,  0, 0, 0, 0,  "no active expert: no group"                       },
        { 8 * expert, 0,      4,  0, 0, 0, 0,  "a zero-byte expert: no group"                     },
    };

    for (const auto & c : group_cases) {
        const moe_pp_batched_f16_groups g = moe_pp_batched_f16_plan_groups(c.slot, c.expert_bytes, c.active, c.cap);
        if (g.experts_per_group != c.per || g.buffers != c.buffers || g.groups != c.groups) {
            std::printf("FAIL: f16 groups for: %s: got %zu x %zu buffers, %zu groups; want %zu x %zu, %zu\n", c.name,
                        g.experts_per_group, g.buffers, g.groups, c.per, c.buffers, c.groups);
            ++failures;
        }
    }
    // Over a grid: the buffers fit in the slot, the groups cover every active expert with none empty, and a slot
    // that holds two experts always splits into halves once there is more than one group.
    for (size_t fit = 1; fit <= 12; ++fit) {
        for (size_t active = 1; active <= 40; ++active) {
            for (size_t cap = 0; cap <= 5; ++cap) {
                const moe_pp_batched_f16_groups g =
                    moe_pp_batched_f16_plan_groups(fit * expert + 7, expert, active, cap);
                const bool ok =
                    g.groups > 0 && g.buffers >= 1 && g.buffers <= 2 && g.buffers * g.experts_per_group <= fit &&
                    g.groups * g.experts_per_group >= active && (g.groups - 1) * g.experts_per_group < active &&
                    (cap == 0 || g.experts_per_group <= cap) && (g.groups == 1 || fit < 2 || g.buffers == 2);
                if (!ok) {
                    std::printf("FAIL: f16 groups break an invariant at fit=%zu active=%zu cap=%zu: %zu x %zu, %zu\n",
                                fit, active, cap, g.experts_per_group, g.buffers, g.groups);
                    ++failures;
                }
            }
        }
    }

    // 12. Where each phase of the f16 arm dequantizes, and whether it waits for the GEMMs that last read its buffer.
    // A slot that holds every active expert (one group) gives each phase its own region at phase.begin experts, so a
    // windowed op's phases are given no wait on an earlier phase's GEMMs; two halves, or the one buffer of a
    // one-expert slot, are reused and keep the wait.
    {
        const size_t weight_elems = 10;  // half-elements per expert
        const size_t f16_expert   = weight_elems * 2;
        const size_t half_elems   = 30;  // one buffer's half-elements: a phase of three experts

        // (A) The slot holds every active expert, no cap: one group, phases at their own regions.
        const moe_pp_batched_f16_groups whole = moe_pp_batched_f16_plan_groups(10 * f16_expert, f16_expert, 7, 0);
        const size_t whole_off0 = moe_pp_batched_phase_weight_offset(whole, 0, 0, half_elems, weight_elems);
        const size_t whole_off3 = moe_pp_batched_phase_weight_offset(whole, 3, 0, half_elems, weight_elems);
        if (whole.groups != 1 || moe_pp_batched_phases_reuse_buffers(whole) || whole_off0 != 0 ||
            whole_off3 != 3 * weight_elems || whole_off0 == whole_off3) {
            std::printf("FAIL: a slot holding every expert: groups %zu reuse %d offsets %zu, %zu; want 1, 0, 0, %zu\n",
                        whole.groups, moe_pp_batched_phases_reuse_buffers(whole) ? 1 : 0, whole_off0, whole_off3,
                        3 * weight_elems);
            ++failures;
        }

        // Two halves (slot of 4, 7 active): two experts per group, so a phase of two experts in a half of
        // 2 * weight_elems, as the executor builds it. Phases alternate buffers 0, 1, 0 and wait for the half's
        // readers.
        const moe_pp_batched_f16_groups halves      = moe_pp_batched_f16_plan_groups(4 * f16_expert, f16_expert, 7, 0);
        const size_t                    halves_half = 2 * weight_elems;
        const size_t                    halves_want[3] = { 0, 20, 0 };
        for (size_t p = 0; p < 3; ++p) {
            const size_t got = moe_pp_batched_phase_weight_offset(halves, p * 2, p % 2, halves_half, weight_elems);
            if (halves.buffers != 2 || halves.experts_per_group != 2 || !moe_pp_batched_phases_reuse_buffers(halves) ||
                got != halves_want[p]) {
                std::printf("FAIL: two halves, phase %zu: buffers %zu x %zu reuse %d offset %zu; want 2 x 2, 1, %zu\n",
                            p, halves.buffers, halves.experts_per_group,
                            moe_pp_batched_phases_reuse_buffers(halves) ? 1 : 0, got, halves_want[p]);
                ++failures;
            }
        }

        // (B) A one-expert slot (3 active): one buffer every phase reuses at offset 0, with the wait.
        const moe_pp_batched_f16_groups single = moe_pp_batched_f16_plan_groups(f16_expert, f16_expert, 3, 0);
        for (size_t p = 0; p < 3; ++p) {
            const size_t got = moe_pp_batched_phase_weight_offset(single, p, 0, weight_elems, weight_elems);
            if (single.buffers != 1 || single.groups != 3 || !moe_pp_batched_phases_reuse_buffers(single) || got != 0) {
                std::printf("FAIL: a one-expert slot, phase %zu: buffers %zu groups %zu reuse %d offset %zu\n", p,
                            single.buffers, single.groups, moe_pp_batched_phases_reuse_buffers(single) ? 1 : 0, got);
                ++failures;
            }
        }

        // A cap that bites (slot of 10, 7 active, cap 2): four groups in two halves, reused.
        const moe_pp_batched_f16_groups capped = moe_pp_batched_f16_plan_groups(10 * f16_expert, f16_expert, 7, 2);
        if (capped.groups != 4 || !moe_pp_batched_phases_reuse_buffers(capped)) {
            std::printf("FAIL: a biting cap: groups %zu reuse %d; want 4, 1\n", capped.groups,
                        moe_pp_batched_phases_reuse_buffers(capped) ? 1 : 0);
            ++failures;
        }

        // Over a grid, building phases as the executor does (at most `window` experts each, buffer p % buffers): every
        // phase's region lies in the slot, and the phases that do not wait write disjoint regions.
        for (size_t fit = 1; fit <= 12; ++fit) {
            for (size_t active = 1; active <= 20; ++active) {
                for (size_t cap = 0; cap <= 3; ++cap) {
                    const size_t                    slot_elems = fit * weight_elems;
                    const moe_pp_batched_f16_groups g =
                        moe_pp_batched_f16_plan_groups(fit * f16_expert + 1, f16_expert, active, cap);
                    for (size_t window = 1; window <= fit; ++window) {
                        const size_t per    = window < g.experts_per_group ? window : g.experts_per_group;
                        const size_t halfs  = per * weight_elems;
                        size_t       prev   = 0;  // end of the previous phase's region, for the disjointness check
                        bool         ok     = true;
                        size_t       nphase = 0;
                        for (size_t begin = 0; begin < active; begin += per, ++nphase) {
                            const size_t end = begin + per < active ? begin + per : active;
                            const size_t off =
                                moe_pp_batched_phase_weight_offset(g, begin, nphase % g.buffers, halfs, weight_elems);
                            const size_t region_end = off + (end - begin) * weight_elems;
                            if (region_end > slot_elems) {
                                ok = false;
                            }
                            if (!moe_pp_batched_phases_reuse_buffers(g)) {
                                ok   = ok && off >= prev;
                                prev = region_end;
                            }
                        }
                        if (!ok) {
                            std::printf("FAIL: f16 phase regions at fit=%zu active=%zu cap=%zu window=%zu\n", fit,
                                        active, cap, window);
                            ++failures;
                        }
                    }
                }
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
