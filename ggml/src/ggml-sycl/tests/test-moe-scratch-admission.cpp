// Planner-owned PP MoE oneDNN scratch admission (llama.cpp-ijla).
//
// Host-only: no SYCL queue, no device, no model, no arena, no unified cache.
// Every shape below is constructed here, which is what makes the cap policy
// testable at all.
//
// The case this port exists for is `over-plan-request-is-refused`: the executor
// used to ask for max(planned, required), so the GPT-OSS layer-0 shape -- one
// 15.8 MiB FP16 expert planned, 32 experts and 506.25 MiB requested -- grew a
// zone that had already been budgeted. A cap that merely clamped would have
// produced a wrong-sized buffer instead; the required behaviour is refusal with
// an attributable reason.

#include "moe-scratch-admission.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using ggml_sycl::pp_moe_onednn_admit_scratch;
using ggml_sycl::pp_moe_onednn_align_scratch_shape;
using ggml_sycl::pp_moe_onednn_checked_align_slot_bytes;
using ggml_sycl::pp_moe_onednn_preflight_scratch;
using ggml_sycl::pp_moe_onednn_reserve_if_admitted;
using ggml_sycl::pp_moe_onednn_scratch_admission;
using ggml_sycl::pp_moe_onednn_scratch_admission_reason;
using ggml_sycl::pp_moe_onednn_scratch_admission_reason_name;
using ggml_sycl::pp_moe_onednn_scratch_shape;
using ggml_sycl::PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void expect_reason(const pp_moe_onednn_scratch_admission & admission,
                          pp_moe_onednn_scratch_admission_reason  expected,
                          const std::string &                     message) {
    check(!admission.allowed && admission.reason == expected,
          message + " (allowed=" + (admission.allowed ? "1" : "0") +
              " reason=" + pp_moe_onednn_scratch_admission_reason_name(admission.reason) + ")");
}

// One 2880x2880 FP16 expert matrix, the activation and output slots that go
// with it, and a single-slot ring. These are the real GPT-OSS layer-0 numbers,
// so the over-plan case below is the shape that motivated the cap.
static pp_moe_onednn_scratch_shape gpt_oss_layer0_plan() {
    pp_moe_onednn_scratch_shape planned;
    planned.weight_slot_bytes     = 2880u * 2880u * 2u;
    planned.activation_slot_bytes = 512u * 2880u * 2u;
    planned.output_slot_bytes     = 512u * 2880u * 4u;
    planned.ring_depth            = 1;
    return planned;
}

static void test_checked_alignment() {
    constexpr size_t max_size = std::numeric_limits<size_t>::max();

    size_t aligned = 0;
    check(pp_moe_onednn_checked_align_slot_bytes(0, &aligned) && aligned == 0, "zero aligns to zero");
    check(pp_moe_onednn_checked_align_slot_bytes(1, &aligned) && aligned == PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT,
          "one byte rounds up to one slot");
    check(pp_moe_onednn_checked_align_slot_bytes(PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT, &aligned) &&
              aligned == PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT,
          "an already-aligned size is unchanged");

    // The whole point of the checked form: the naive (value + 255) / 256 * 256
    // wraps here, and a wrapped size compares as SMALL, which would admit an
    // unrepresentable request.
    aligned = 12345;
    check(!pp_moe_onednn_checked_align_slot_bytes(max_size, &aligned), "an unrepresentable size fails to align");
    check(aligned == 12345, "a failed alignment leaves the output untouched");

    const size_t largest_alignable = max_size - (PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT - 1);
    check(pp_moe_onednn_checked_align_slot_bytes(largest_alignable, &aligned),
          "the largest alignable size still aligns");

    pp_moe_onednn_scratch_shape shape;
    shape.weight_slot_bytes     = 1;
    shape.activation_slot_bytes = 1;
    shape.output_slot_bytes     = max_size;
    shape.ring_depth            = 1;
    check(!pp_moe_onednn_align_scratch_shape(shape, nullptr), "one unalignable dimension fails the whole shape");
}

static void test_exact_and_smaller_requests_are_admitted() {
    const pp_moe_onednn_scratch_shape planned = gpt_oss_layer0_plan();

    const auto exact = pp_moe_onednn_admit_scratch(planned, planned);
    check(exact.allowed && exact.reason == pp_moe_onednn_scratch_admission_reason::ALLOWED,
          "a request equal to the plan is admitted");

    pp_moe_onednn_scratch_shape smaller = planned;
    smaller.weight_slot_bytes -= PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    smaller.activation_slot_bytes -= PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    smaller.output_slot_bytes -= PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    const auto under = pp_moe_onednn_admit_scratch(planned, smaller);
    check(under.allowed, "a request under the plan is admitted");

    // The reservation must be told to allocate the ALIGNED size, not the raw
    // request; handing back the raw one would under-allocate the slot the
    // caller was just told it could use.
    pp_moe_onednn_scratch_shape ragged = planned;
    ragged.weight_slot_bytes -= 1;
    pp_moe_onednn_scratch_shape aligned_required;
    const auto                  ragged_admission = pp_moe_onednn_preflight_scratch(planned, ragged, &aligned_required);
    check(ragged_admission.allowed, "a ragged under-plan request is admitted");
    check(aligned_required.weight_slot_bytes == planned.weight_slot_bytes,
          "the admitted shape is the aligned request, not the raw one");
    check(aligned_required.ring_depth == ragged.ring_depth, "the admitted ring depth is the requested one");
}

// The deliberate correction to the salvaged branch. There the aligned request
// was compared against the RAW plan, so any plan whose slot sizes were not a
// multiple of 256 refused a request equal to itself -- and the cache's own
// reservation, which passes the plan verbatim, was exactly such a request. The
// cap would have failed the whole PP MoE oneDNN path closed for a rounding
// artifact.
static void test_an_unaligned_plan_admits_itself() {
    pp_moe_onednn_scratch_shape planned = gpt_oss_layer0_plan();
    planned.weight_slot_bytes -= 1;
    planned.activation_slot_bytes -= 17;
    planned.output_slot_bytes -= 255;

    const auto self = pp_moe_onednn_admit_scratch(planned, planned);
    check(self.allowed, "an unaligned plan admits a request equal to itself");

    // The ceiling is still the aligned plan, not an open door: one alignment
    // unit past it is refused.
    pp_moe_onednn_scratch_shape over = planned;
    over.weight_slot_bytes += PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    expect_reason(pp_moe_onednn_admit_scratch(planned, over), pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,
                  "an unaligned plan still caps at its aligned ceiling");
}

static void test_missing_plan_fails_closed() {
    const pp_moe_onednn_scratch_shape request = gpt_oss_layer0_plan();

    const char * labels[] = { "weight", "activation", "output", "ring depth" };
    for (int i = 0; i < 4; ++i) {
        pp_moe_onednn_scratch_shape planned = request;
        switch (i) {
            case 0:
                planned.weight_slot_bytes = 0;
                break;
            case 1:
                planned.activation_slot_bytes = 0;
                break;
            case 2:
                planned.output_slot_bytes = 0;
                break;
            default:
                planned.ring_depth = 0;
                break;
        }
        expect_reason(pp_moe_onednn_admit_scratch(planned, request),
                      pp_moe_onednn_scratch_admission_reason::MISSING_PLAN,
                      std::string("a zero planned ") + labels[i] + " fails closed");
    }

    // An unplanned device is the common case, not an exotic one: a dense model
    // publishes nothing here at all.
    const pp_moe_onednn_scratch_shape unplanned;
    expect_reason(pp_moe_onednn_admit_scratch(unplanned, request), pp_moe_onednn_scratch_admission_reason::MISSING_PLAN,
                  "an entirely unpublished plan fails closed");

    // A plan too large to align is not a usable ceiling either.
    pp_moe_onednn_scratch_shape unalignable = request;
    unalignable.weight_slot_bytes           = std::numeric_limits<size_t>::max();
    expect_reason(pp_moe_onednn_admit_scratch(unalignable, request),
                  pp_moe_onednn_scratch_admission_reason::MISSING_PLAN, "an unalignable plan fails closed");
}

static void test_invalid_requests_are_rejected() {
    const pp_moe_onednn_scratch_shape planned = gpt_oss_layer0_plan();

    const char * labels[] = { "weight", "activation", "output", "ring depth" };
    for (int i = 0; i < 4; ++i) {
        pp_moe_onednn_scratch_shape required = planned;
        switch (i) {
            case 0:
                required.weight_slot_bytes = 0;
                break;
            case 1:
                required.activation_slot_bytes = 0;
                break;
            case 2:
                required.output_slot_bytes = 0;
                break;
            default:
                required.ring_depth = 0;
                break;
        }
        expect_reason(pp_moe_onednn_admit_scratch(planned, required),
                      pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST,
                      std::string("a zero required ") + labels[i] + " is rejected");
    }

    // Overflow is reported as an invalid REQUEST even when the plan is equally
    // unrepresentable, so the caller is told which side it controls.
    const size_t                max_size = std::numeric_limits<size_t>::max();
    pp_moe_onednn_scratch_shape overflow_plan;
    overflow_plan.weight_slot_bytes     = max_size;
    overflow_plan.activation_slot_bytes = max_size;
    overflow_plan.output_slot_bytes     = max_size;
    overflow_plan.ring_depth            = 1;
    for (int i = 0; i < 3; ++i) {
        pp_moe_onednn_scratch_shape required = planned;
        switch (i) {
            case 0:
                required.weight_slot_bytes = max_size;
                break;
            case 1:
                required.activation_slot_bytes = max_size;
                break;
            default:
                required.output_slot_bytes = max_size;
                break;
        }
        expect_reason(pp_moe_onednn_admit_scratch(overflow_plan, required),
                      pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST,
                      std::string("an unalignable required dimension ") + labels[i] + " is rejected");
    }
}

static void test_over_plan_request_is_refused() {
    const pp_moe_onednn_scratch_shape planned = gpt_oss_layer0_plan();

    // The motivating shape: 32 experts against a one-expert plan. 506.25 MiB of
    // FP16 weights is 32x the planned slot, and it must be refused rather than
    // enlarging the ring.
    pp_moe_onednn_scratch_shape thirty_two_experts = planned;
    thirty_two_experts.weight_slot_bytes           = planned.weight_slot_bytes * 32;
    const auto refused                             = pp_moe_onednn_admit_scratch(planned, thirty_two_experts);
    expect_reason(refused, pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,
                  "a 32-expert batch against a one-expert plan is refused");
    check(std::strcmp(pp_moe_onednn_scratch_admission_reason_name(refused.reason), "weight-cap") == 0,
          "the refusal is attributable by name");

    // One alignment unit past the plan is the boundary. Anything that admits
    // here is not a cap.
    struct {
        size_t pp_moe_onednn_scratch_shape::*  byte_field;
        pp_moe_onednn_scratch_admission_reason expected;
        const char *                           label;
    } byte_cases[] = {
        { &pp_moe_onednn_scratch_shape::weight_slot_bytes,     pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,
         "weight"     },
        { &pp_moe_onednn_scratch_shape::activation_slot_bytes, pp_moe_onednn_scratch_admission_reason::ACTIVATION_CAP,
         "activation" },
        { &pp_moe_onednn_scratch_shape::output_slot_bytes,     pp_moe_onednn_scratch_admission_reason::OUTPUT_CAP,
         "output"     },
    };

    for (const auto & byte_case : byte_cases) {
        pp_moe_onednn_scratch_shape required = planned;
        required.*byte_case.byte_field += PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
        expect_reason(pp_moe_onednn_admit_scratch(planned, required), byte_case.expected,
                      std::string("one slot past the planned ") + byte_case.label + " is refused");
    }

    pp_moe_onednn_scratch_shape deeper_ring = planned;
    deeper_ring.ring_depth                  = planned.ring_depth + 1;
    expect_reason(pp_moe_onednn_admit_scratch(planned, deeper_ring), pp_moe_onednn_scratch_admission_reason::RING_CAP,
                  "a deeper ring than planned is refused");

    // Sub-alignment overrun. A cap that compared raw bytes would admit this,
    // and the slot the caller then wrote past is 256 bytes, not one.
    pp_moe_onednn_scratch_shape one_byte_over = planned;
    one_byte_over.weight_slot_bytes += 1;
    expect_reason(pp_moe_onednn_admit_scratch(planned, one_byte_over),
                  pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,
                  "a request one byte past the plan is refused at slot granularity");
}

static void test_reason_names_are_stable() {
    const struct {
        pp_moe_onednn_scratch_admission_reason reason;
        const char *                           name;
    } expected[] = {
        { pp_moe_onednn_scratch_admission_reason::ALLOWED,         "allowed"         },
        { pp_moe_onednn_scratch_admission_reason::MISSING_PLAN,    "missing-plan"    },
        { pp_moe_onednn_scratch_admission_reason::INVALID_REQUEST, "invalid-request" },
        { pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,      "weight-cap"      },
        { pp_moe_onednn_scratch_admission_reason::ACTIVATION_CAP,  "activation-cap"  },
        { pp_moe_onednn_scratch_admission_reason::OUTPUT_CAP,      "output-cap"      },
        { pp_moe_onednn_scratch_admission_reason::RING_CAP,        "ring-cap"        },
    };

    for (const auto & entry : expected) {
        check(std::strcmp(pp_moe_onednn_scratch_admission_reason_name(entry.reason), entry.name) == 0,
              std::string("reason name is stable: ") + entry.name);
    }

    // The reasons reach logs and the source-contract gate greps for them, so a
    // renamed enumerator must not silently collapse onto another name.
    for (const auto & outer : expected) {
        for (const auto & inner : expected) {
            if (outer.reason == inner.reason) {
                continue;
            }
            check(std::strcmp(pp_moe_onednn_scratch_admission_reason_name(outer.reason),
                              pp_moe_onednn_scratch_admission_reason_name(inner.reason)) != 0,
                  "two reasons must not share a name");
        }
    }

    // A default-constructed verdict must read as a refusal, so a caller that
    // never ran the preflight cannot mistake it for an admission.
    const pp_moe_onednn_scratch_admission unset;
    check(!unset.allowed && unset.reason == pp_moe_onednn_scratch_admission_reason::MISSING_PLAN,
          "an unevaluated admission defaults to refused");
}

// The behavioural half of the fail-closed property: a refused request must
// reach NO side effect. Source order can show where the preflight sits; only
// this can show that nothing downstream of it ran.
static void test_refusal_reaches_no_side_effect() {
    const pp_moe_onednn_scratch_shape planned = gpt_oss_layer0_plan();

    struct gate_result {
        bool                            reserved = false;
        pp_moe_onednn_scratch_admission admission;
        uint32_t                        side_effect_count = 0;
        pp_moe_onednn_scratch_shape     side_effect_shape;
    };

    const auto run_gate = [&](const pp_moe_onednn_scratch_shape & requested) {
        gate_result result;
        result.reserved = pp_moe_onednn_reserve_if_admitted(planned, requested, result.admission,
                                                            [&](const pp_moe_onednn_scratch_shape & admitted) {
                                                                ++result.side_effect_count;
                                                                result.side_effect_shape = admitted;
                                                                return true;
                                                            });
        return result;
    };

    // Positive control. Without it the zero-count assertions below would pass
    // on a seam that never invokes its callback at all.
    const gate_result admitted = run_gate(planned);
    check(admitted.reserved && admitted.admission.allowed, "an admitted request reserves");
    check(admitted.side_effect_count == 1, "an admitted request reaches the side effect exactly once");
    check(admitted.side_effect_shape.weight_slot_bytes == planned.weight_slot_bytes,
          "the side effect receives the admitted shape");

    pp_moe_onednn_scratch_shape over_cap = planned;
    over_cap.weight_slot_bytes += PP_MOE_ONEDNN_SCRATCH_SLOT_ALIGNMENT;
    const gate_result refused = run_gate(over_cap);
    check(!refused.reserved, "an over-cap request does not reserve");
    expect_reason(refused.admission, pp_moe_onednn_scratch_admission_reason::WEIGHT_CAP,
                  "an over-cap request is refused for the right reason");
    check(refused.side_effect_count == 0, "an over-cap request reaches no lock and no allocator");

    pp_moe_onednn_scratch_shape zero = planned;
    zero.weight_slot_bytes           = 0;
    const gate_result zero_result    = run_gate(zero);
    check(!zero_result.reserved && zero_result.side_effect_count == 0,
          "a zero-dimension request reaches no lock and no allocator");

    pp_moe_onednn_scratch_shape unalignable = planned;
    unalignable.weight_slot_bytes           = std::numeric_limits<size_t>::max();
    const gate_result unalignable_result    = run_gate(unalignable);
    check(!unalignable_result.reserved && unalignable_result.side_effect_count == 0,
          "an unalignable request reaches no lock and no allocator");

    // A reservation that fails for its own reasons must be reported as a
    // failure, not laundered into success by the admission having passed.
    pp_moe_onednn_scratch_admission admission;
    const bool                      failed_reserve = pp_moe_onednn_reserve_if_admitted(
        planned, planned, admission, [](const pp_moe_onednn_scratch_shape &) { return false; });
    check(admission.allowed && !failed_reserve, "an admitted request that fails to allocate still returns false");
}

int main() {
    const struct {
        const char * name;
        void (*fn)();
    } cases[] = {
        { "checked-alignment",                       test_checked_alignment                       },
        { "exact-and-smaller-requests-are-admitted", test_exact_and_smaller_requests_are_admitted },
        { "an-unaligned-plan-admits-itself",         test_an_unaligned_plan_admits_itself         },
        { "missing-plan-fails-closed",               test_missing_plan_fails_closed               },
        { "invalid-requests-are-rejected",           test_invalid_requests_are_rejected           },
        { "over-plan-request-is-refused",            test_over_plan_request_is_refused            },
        { "reason-names-are-stable",                 test_reason_names_are_stable                 },
        { "refusal-reaches-no-side-effect",          test_refusal_reaches_no_side_effect          },
    };

    for (const auto & test_case : cases) {
        try {
            test_case.fn();
        } catch (const std::exception & e) {
            std::cout << "FAIL: " << test_case.name << ": " << e.what() << "\n";
            return 1;
        }
        std::cout << "ok: " << test_case.name << "\n";
    }

    std::cout << "test-moe-scratch-admission: PASS (" << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
    return 0;
}
