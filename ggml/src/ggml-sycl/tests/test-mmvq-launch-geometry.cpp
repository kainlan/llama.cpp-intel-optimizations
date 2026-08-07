// MMVQ sub-group-per-row launch geometry (llama.cpp-99ke).
//
// Host-only: no SYCL queue, no device, no model. The geometry is pure integer
// arithmetic on the host side of the dispatch, which is exactly why it can be
// proven here -- the GPU-side symptom (an nd_range the driver refuses with
// "Non-uniform work-groups are not supported by the target device") is a
// consequence of numbers computed before anything is submitted.
//
// The case this fix exists for is `nrows=1`: `ceil_div(nrows, GGML_SYCL_MMV_Y)`
// is a no-op because GGML_SYCL_MMV_Y is 1, so a one-row slice launched a global
// range of 1*WARP_SIZE against a work-group of 16*WARP_SIZE and threw at the
// dispatcher. `slice-of-one-pads-to-one-workgroup` below is that shape.
//
// The other load-bearing case is `already-aligned-shapes-are-unchanged`: every
// currently-passing shape must launch byte-identical geometry, or this fix is a
// performance change wearing a bug fix's clothes.

#include "mmvq-launch-geometry.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using ggml_sycl::mmvq_launch_covers_rows;
using ggml_sycl::mmvq_launch_is_uniform;
using ggml_sycl::mmvq_pad_rows_to_workgroups;

// The value every one of the 11 mmvq.cpp launch sites uses.
static constexpr int SUBGROUPS = 16;

static void check(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The whole contract in one place: a padded count must be launchable (uniform
// work-groups) and must not lose a row. Asserting both together is what keeps a
// degenerate "return 0" or "return rows" from passing half the property.
static void check_launchable(const int rows, const std::string & label) {
    const int padded = mmvq_pad_rows_to_workgroups(rows, SUBGROUPS);
    check(mmvq_launch_is_uniform(padded, SUBGROUPS),
          label + ": padded " + std::to_string(padded) + " rows is not a whole number of work-groups");
    check(mmvq_launch_covers_rows(padded, rows),
          label + ": padded " + std::to_string(padded) + " rows does not cover " + std::to_string(rows));
}

static void test_slice_of_one_pads_to_one_workgroup() {
    // test-ggml-sycl-soa Test 10 (32x1) and test-dmmv-q6k-coalesced's first case
    // (ne=[8192,1,1,1]) are both this shape.
    check(mmvq_pad_rows_to_workgroups(1, SUBGROUPS) == 16, "one row pads to one full work-group");
    check_launchable(1, "one row");
}

static void test_already_aligned_shapes_are_unchanged() {
    // Every shape that passes today. Padding these must be the identity, or the
    // fix silently changes the geometry of the hot path.
    const int aligned[] = { 0, 16, 32, 64, 512, 4096, 14336, 32000 };
    for (const int rows : aligned) {
        check(mmvq_pad_rows_to_workgroups(rows, SUBGROUPS) == rows,
              "aligned row count " + std::to_string(rows) + " must pad to itself");
        check_launchable(rows, "aligned " + std::to_string(rows));
    }
}

static void test_unaligned_shapes_round_up_to_the_next_workgroup() {
    // Test 9 (64x64) already passed; 65 is the smallest slice that did not.
    check(mmvq_pad_rows_to_workgroups(17, SUBGROUPS) == 32, "17 rows pads to 32");
    check(mmvq_pad_rows_to_workgroups(31, SUBGROUPS) == 32, "31 rows pads to 32");
    check(mmvq_pad_rows_to_workgroups(65, SUBGROUPS) == 80, "65 rows pads to 80");
}

static void test_every_slice_up_to_a_thousand_is_launchable() {
    // The defect's real condition is `nrows % 16 != 0`, not `nrows == 1`, and a
    // row slice is `row_high - row_low` -- an arbitrary integer. Sweeping the
    // range is the only check that would have caught the original bug without
    // knowing which slice sizes split-tensor support happens to produce.
    for (int rows = 0; rows <= 1000; ++rows) {
        check_launchable(rows, "slice of " + std::to_string(rows));
        // Padding is a rounding step, not a resize: it must never overshoot by a
        // whole work-group, or a slice of 16 would launch 32 rows of nothing.
        const int padded = mmvq_pad_rows_to_workgroups(rows, SUBGROUPS);
        check(padded - rows < SUBGROUPS, "slice of " + std::to_string(rows) + " overshot to " + std::to_string(padded));
    }
}

static void test_degenerate_inputs_launch_nothing_extra() {
    // An empty slice must stay empty: padding 0 up to a full work-group would
    // submit 16 rows that the kernel's `row >= nrows` guard rejects one by one,
    // turning "no work" into a launch.
    check(mmvq_pad_rows_to_workgroups(0, SUBGROUPS) == 0, "an empty slice launches nothing");
    check(mmvq_pad_rows_to_workgroups(-1, SUBGROUPS) == 0, "a negative slice launches nothing");
    // A one-subgroup work-group is uniform for any row count already.
    check(mmvq_pad_rows_to_workgroups(7, 1) == 7, "a single-subgroup work-group needs no padding");
}

static void test_the_uniformity_predicate_can_fail() {
    // A positive control for check_launchable: `mmvq_launch_is_uniform` must be
    // capable of returning false, or every "is launchable" assertion above is
    // decorative. 1 row against 16 sub-groups is the exact geometry that throws.
    check(!mmvq_launch_is_uniform(1, SUBGROUPS), "an unpadded single row is not uniform");
    check(!mmvq_launch_is_uniform(65, SUBGROUPS), "an unpadded 65-row slice is not uniform");
    check(!mmvq_launch_covers_rows(16, 17), "16 padded rows cannot cover 17");
}

int main() {
    const struct {
        const char * name;
        void (*fn)();
    } cases[] = {
        { "slice-of-one-pads-to-one-workgroup",         test_slice_of_one_pads_to_one_workgroup              },
        { "already-aligned-shapes-are-unchanged",       test_already_aligned_shapes_are_unchanged            },
        { "unaligned-shapes-round-up",                  test_unaligned_shapes_round_up_to_the_next_workgroup },
        { "every-slice-up-to-a-thousand-is-launchable", test_every_slice_up_to_a_thousand_is_launchable      },
        { "degenerate-inputs-launch-nothing-extra",     test_degenerate_inputs_launch_nothing_extra          },
        { "the-uniformity-predicate-can-fail",          test_the_uniformity_predicate_can_fail               },
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

    std::cout << "test-mmvq-launch-geometry: PASS (" << (sizeof(cases) / sizeof(cases[0])) << " cases)\n";
    return 0;
}
